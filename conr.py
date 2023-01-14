import torch

from model.cinn import CINN
from model.decoder_small import RGBADecoderNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def UDPClip(x):
    return torch.clamp(x, min=0, max=1)


class CoNR():
    def __init__(self, args):
        self.args = args
        self.cinnnet = CINN(4+3+4+64, 64)
        self.rgbadecodernet = RGBADecoderNet()
        self.udpadecodernet = RGBADecoderNet()
        self.reset_charactersheet()
        self.device()

    def dist(self):
        args = self.args
        if args.distributed:
            self.cinnnet = torch.nn.parallel.DistributedDataParallel(
                self.cinnnet,
                device_ids=[
                    args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False
            )
            self.udpadecodernet = torch.nn.parallel.DistributedDataParallel(
                self.udpadecodernet,
                device_ids=[
                    args.local_rank],
                output_device=args.local_rank,
            )
            self.rgbadecodernet = torch.nn.parallel.DistributedDataParallel(
                self.rgbadecodernet,
                device_ids=[
                    args.local_rank],
                output_device=args.local_rank,
            )

    def load_model(self, path):
        self.cinnnet.load_state_dict(
            torch.load('{}/cinnnet.pth'.format(path), map_location=device))
        self.udpadecodernet.load_state_dict(
            torch.load('{}/udpadecodernet.pth'.format(path), map_location=device))
        self.rgbadecodernet.load_state_dict(
            torch.load('{}/rgbadecodernet.pth'.format(path), map_location=device))
        self.optimizer_path = '{}/optimizer.pth'.format(path)

    def train(self):
        self.cinnnet.train()
        self.udpadecodernet.train()
        self.rgbadecodernet.train()

    def eval(self):
        self.cinnnet.eval()
        self.udpadecodernet.eval()
        self.rgbadecodernet.eval()

    def device(self):
        self.cinnnet.to(device)
        self.udpadecodernet.to(device)
        self.rgbadecodernet.to(device)

    def data_to_device(self, data):

        with torch.cuda.amp.autocast(enabled=False):
            for name in ["character_labels",  "pose_label"]:
                if name in data:
                    data[name] = data[name].to(
                        device, non_blocking=False).float()
            for name in ["pose_images", "pose_mask", "character_images", "character_masks"]:
                if name in data:
                    data[name] = data[name].to(
                        device, non_blocking=False).float() / 255.0
            if "pose_images" in data:
                data["num_pose_images"] = data["pose_images"].shape[1]
                data["num_samples"] = data["pose_images"].shape[0]
            if "character_images" in data:
                data["num_character_images"] = data["character_images"].shape[1]
                data["num_samples"] = data["character_images"].shape[0]
            if "pose_images" in data and "character_images" in data:
                assert (data["pose_images"].shape[0] ==
                        data["character_images"].shape[0])
        return data

    def reset_charactersheet(self):
        self.parser_ckpt = None

    def model_step(self, data):
        self.eval()
        with torch.cuda.amp.autocast(enabled=False):
            pred = {}
            if self.parser_ckpt:
                pred["parser"] = self.parser_ckpt
            else:
                pred = self.character_parser_forward(data, pred)
                self.parser_ckpt = pred["parser"]

            pred = self.pose_parser_sc_forward(data, pred)
            for stage in range(1, -1, -1):
                pred = self.shader_forward(data, pred, stage=stage)
        return pred

    def shader_forward(self, data, pred, stage=0):
        if stage == 0:
            shader_stage = "shader"
        else:
            shader_stage = "shader_{}".format(stage)
        shader_stage_last = "shader_{}".format(stage+1)
        pred[shader_stage] = {}
        shader_target_sudp = pred["pose_parser"]["pred"][:, 0:3, :,
                                                         :] if "pose_parser" in pred and "pred" in pred["pose_parser"] else None
        shader_target_a = None
        if "pose_mask" in data:
            shader_target_a = data["pose_mask"]
        if "pose_label" in data:
            shader_target_sudp = data["pose_label"][:, :3, :, :]
        if shader_target_a is None:
            shader_target_a = pred["pose_parser"]["pred"][:, 3:4, :, :]

        if shader_stage_last in pred:
            x_target_sudp_a = pred[shader_stage_last]["y_weighted_warp_decoded_rgba"]
        else:
            x_target_sudp_a = torch.cat((
                shader_target_sudp*shader_target_a,
                shader_target_a
            ), 1)
        
        pred[shader_stage].update({
            "x_target_sudp_a": x_target_sudp_a
        })

        assert ("num_character_images" in data), "ERROR: No Character Sheet input."

        character_images_rgb_nmchw, num_character_images = data[
            "character_images"], data["num_character_images"]
        
        character_a_nmchw = data["character_masks"]
        if torch.any(torch.mean(character_a_nmchw, (0, 2, 3, 4)) > 0.95):
            raise ValueError(
                "No transparent area found in the image, PLEASE separate the foreground of input character sheets.")
        shader_character_sudp_nmchw = pred["parser"]["pred"][:, :, 0:3, :, :]
        x_reference_rgb_a_sudp = torch.cat([character_a_nmchw[:, :, :, :, :] *
                                            character_images_rgb_nmchw[:,
                                                                       :, :, :, :],
                                            character_a_nmchw[:, :, :, :, :],
                                            shader_character_sudp_nmchw[:,
                                                                        :, :, :, :]
                                            ], 2)
        assert (x_reference_rgb_a_sudp.dtype == torch.float32)
        assert (x_reference_rgb_a_sudp.shape[0] ==
                character_images_rgb_nmchw.shape[0])
        assert (x_reference_rgb_a_sudp.shape[1] == num_character_images)

        pred[shader_stage].update({
            "x_reference_rgb_a_sudp": x_reference_rgb_a_sudp,
        })
        if shader_stage_last in pred:
            from_last = pred[shader_stage_last]["y_msg"]
        else:
            from_last = pred["parser"]["encoder"]

        retdic = self.cinnnet(x_target_sudp_a.detach(), torch.cat(
            (x_reference_rgb_a_sudp, from_last), dim=2))
        pred[shader_stage]["y_msg"] = retdic
        assert (retdic.shape[2] == 64), retdic.shape
        
        cont = torch.softmax(retdic[:, :, 0:1, :, :], dim=1)
        y_weighted_msg = torch.sum(
                retdic * cont, dim=1)

        dec_out = self.rgbadecodernet(y_weighted_msg)
        y_weighted_RGB = dec_out[:, 0:3, :, :]
        y_weighted_A = dec_out[:, 3:4, :, :]

        y_weighted_warp_decoded_rgba = torch.cat(
            (y_weighted_RGB*y_weighted_A, y_weighted_A), dim=1
        )
        assert(y_weighted_warp_decoded_rgba.shape[1] == 4)
        assert(
            y_weighted_warp_decoded_rgba.shape[-1] == character_images_rgb_nmchw.shape[-1])
        
        pred[shader_stage]["y_weighted_warp_decoded_rgba"] = y_weighted_warp_decoded_rgba

        return pred

    def character_parser_forward(self, data, pred):
        if not("num_character_images" in data and "character_images" in data):
            return pred
        pred["parser"] = {"pred": None}

        inputs_rgb_nmchw, labels_a_nmchw, num_samples, num_character_images = data[
            "character_images"],  data["character_masks"], data["num_samples"], data["num_character_images"]
        target_udp = torch.zeros(
            (inputs_rgb_nmchw.shape[0], 4, inputs_rgb_nmchw.shape[3], inputs_rgb_nmchw.shape[4]), device=device)
        src_sudp_feat = torch.zeros(
            (inputs_rgb_nmchw.shape[0], inputs_rgb_nmchw.shape[1], 3+64, inputs_rgb_nmchw.shape[3], inputs_rgb_nmchw.shape[4]), device=device)
        encoder_out = self.cinnnet(target_udp,  torch.cat(
            (inputs_rgb_nmchw, labels_a_nmchw, src_sudp_feat), dim=2))
        udp_out_fchw = self.udpadecodernet(encoder_out.view(
            (num_samples * num_character_images, encoder_out.shape[2], encoder_out.shape[3], encoder_out.shape[4])))

        udp_out = udp_out_fchw.view(
            (num_samples, num_character_images, udp_out_fchw.shape[1], udp_out_fchw.shape[2], udp_out_fchw.shape[3]))

        pred["parser"]["pred"] = UDPClip(udp_out)
        pred["parser"]["encoder"] = encoder_out
        return pred

    def pose_parser_sc_forward(self, data, pred):
        if not("num_pose_images" in data and "pose_images" in data):
            return pred
        inputs_aug_rgb_nmchw, inputs_label_a_nchw, num_samples, num_pose_images = data[
            "pose_images"],  data["pose_mask"], data["num_samples"], data["num_pose_images"]

        src_sudp_feat = torch.zeros(
            (inputs_aug_rgb_nmchw.shape[0], inputs_aug_rgb_nmchw.shape[1], 3+64, inputs_aug_rgb_nmchw.shape[3], inputs_aug_rgb_nmchw.shape[4]), device=device)
        inputs_label_a_nmchw = inputs_label_a_nchw.unsqueeze(1).repeat(
            [1,  inputs_aug_rgb_nmchw.shape[1], 1, 1, 1])
        reshaped_inputs = torch.cat(
            (inputs_aug_rgb_nmchw, inputs_label_a_nmchw, src_sudp_feat), dim=2)
        reshaped_inputs = reshaped_inputs.reshape(
            (num_samples * num_pose_images, 1, reshaped_inputs.shape[2], reshaped_inputs.shape[3], reshaped_inputs.shape[4]))
        target_udp = torch.zeros(
            (reshaped_inputs.shape[0], 4, inputs_aug_rgb_nmchw.shape[3], inputs_aug_rgb_nmchw.shape[4]), device=device)
        encoder_out = self.cinnnet(target_udp, reshaped_inputs)
        udp_out_fchw = self.udpadecodernet(encoder_out.view(
            (num_samples * num_pose_images, encoder_out.shape[2], encoder_out.shape[3], encoder_out.shape[4])))
        udp_out = udp_out_fchw.view(
            (num_samples, num_pose_images, udp_out_fchw.shape[1], udp_out_fchw.shape[2], udp_out_fchw.shape[3]))
        pred["pose_parser"] = {}
        pred["pose_parser"]["sc_preds"] = UDPClip(udp_out)

        pred["pose_parser"]["pred"] = UDPClip(
            udp_out[:, 0, :, :, :])

        return pred
