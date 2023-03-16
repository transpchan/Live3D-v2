# Copyright 2023 (c) Live3D-v2 transpchan.

# ONLY USE THIS GUI FOR Live3D-v2. 

import gradio as gr
import os
import sys
import shutil
from pathlib import Path
import subprocess
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
if __name__ == "__main__":
    langdata = {
        "NO_CHAR_SHEET": ["## Please select at least two character design sheets", "## 请选择至少两张角色设计图"],
        "NO_UDP": ["## MMD2UDP plugin fails to run, check if your .vmd or .pmx files are correct.", "## MMD2UDP插件运行失败，请检查您的.vmd或.pmx文件是否正确"],
        "NO_PMX": ["## Please select a .zip file containing PMX or FBX mesh", "## 请选择一个包含 PMX或FBX模型 的.zip 文件"],
        "MMD2UDP_FAIL": ["## MMD2UDP plugin fails to run, check if your .vmd or .pmx files are correct.", "## MMD2UDP插件运行失败，请检查您的.vmd或.pmx文件是否正确"],
        "GEN_FAIL": ["## An error occurs during video generation process. Please post the full command line to https://github.com/transpchan/Live3D-v2/issues.", "## 在视频生成过程中发生错误。请将完整的命令行发布到https://github.com/transpchan/Live3D-v2/issues."],
        "CONV_FAIL": ["## An error occurs during video conversion process. Please post the full command line to https://github.com/transpchan/Live3D-v2/issues.", "## 在视频转换过程中发生错误。请将完整的命令行发布到 https://github.com/transpchan/Live3D-v2/issues."],
        "DONE": ["## Done! There is also a output_adobe_premiere.mov with transparent background in the folder. Please include our link github.com/transpchan/Live3D-v2 when sharing the video.", "## 完成！文件夹中还有一个output_adobe_premiere.mov，是透明背景的视频。根据中国大陆的AIGC相关法律法规，本程序生成视频需要在醒目位置增加水印。使用者检查生成结果并对其负责"],
        "SELECT_CHAR_SHEET": ["Select at least two .PNG files with transparent background", "选择至少两张透明背景的.PNG文件"],
        "SELECT_PMX": ["Select a .zip file containing a MMD mesh (.pmx) file and texture files", "选择一个包含MMD模型(.pmx)文件和纹理文件的.zip文件"],
        "SELECT_MOTION": ["Select a MMD motion file .vmd", "选择一个MMD动作文件.vmd"],
        "SELECT_CAMERA": ["Select a MMD camera file .vmd (Optional)", "选择一个MMD相机文件.vmd（可选）"],
        "RUN": ["Run", "运行"],
        "EXAMPLE": ["Examples (If you can't provide any of the files above, click one of these.)", " 示例（如果您无法提供上述任何文件，请单击其中一个。）"],
        "UDP_POINTCLOUD": ["UDP Pointcloud Generation", "UDP点云生成"],
        "UDP_POINTCLOUD_DESC": ["## UDP Pointcloud Generation from set of character design images", "## 从一组角色设计图生成UDP点云"],
        "VIDEO_GEN": ["Video Generation", "视频生成"],
        "VIDEO_GEN_DESC": ["## Animate your hand-drawn character with a MikuMikuDance-covnerted UDP sequence", "## 使用MikuMikuDance转换的UDP序列为手绘角色添加动作"],
        "COLORIZE_MESH": ["Colorize Mesh", "模型上色"],
        "COLORIZE_MESH_DESC": ["## Colorize mesh (change skin) using a set of character design sheets. ", "## 使用一组角色设计图为模型上色（更改皮肤）。"],
    }

    if len(sys.argv) > 1 and sys.argv[1] == "zh":
        languageid = 1
    else:
        languageid = 0
    Lang = {}
    for each in langdata:
        Lang[each] = langdata[each][languageid]

    def conr_fn(examples=None, character_sheets=None,  mmd_pmx=None, mmd_motion=None, mmd_camera=None):
        if (examples is None or examples == "") and (character_sheets is None or len(character_sheets) < 2):
            return Lang["NO_CHAR_SHEET"], None

        if Path("character_sheet").exists():
            shutil.rmtree(Path("character_sheet"))
        if Path("./MMD2UDP/output").exists():
            shutil.rmtree(Path("./MMD2UDP/output"))
        if Path("results").exists():
            shutil.rmtree(Path("results"))
        Path("character_sheet").mkdir(parents=True, exist_ok=True)
        Path("./MMD2UDP/output").mkdir(parents=True, exist_ok=True)
        Path("results").mkdir(parents=True, exist_ok=True)
        if examples is not None and examples != "":
            poses = Path(examples)
            character = Path(examples+"_images")

        else:
            for i, e in enumerate(character_sheets):
                with open(f"character_sheet/{i}.png", "wb") as f:
                    e.seek(0)
                    f.write(e.read())
                    e.seek(0)
            poses = Path("./MMD2UDP/output")
            character = Path("./character_sheet")
            if mmd_pmx is not None:
                with open("./MMD2UDP/model.zip", "wb") as f:
                    mmd_pmx.seek(0)
                    f.write(mmd_pmx.read())
                    mmd_pmx.seek(0)
            else:
                return Lang["NO_PMX"], None
            if mmd_motion is not None:
                with open("./MMD2UDP/motion.vmd", "wb") as f:
                    mmd_motion.seek(0)
                    f.write(mmd_motion.read())
                    mmd_motion.seek(0)
           
            if mmd_camera is not None:
                with open("./MMD2UDP/camera.vmd", "wb") as f:
                    mmd_camera.seek(0)
                    f.write(mmd_camera.read())
                    mmd_camera.seek(0)
            cwd = os.getcwd()
            os.chdir("MMD2UDP")
            if os.name == "nt":
                os.system("UltraDensePose.exe")
            else:
                os.system("udp")
            os.chdir(cwd)
        if len(list(poses.glob("*"))) == 0:
            return Lang["NO_UDP"], None

        else:
            ret = os.system(
                "python train.py --test_input_poses_images={} --test_input_person_images={}".format(poses, character))

            if ret != 0:
                return Lang["GEN_FAIL"], None
            torun = 'ffmpeg -r 30 -y -i ./results/%d.png -i watermark.png -filter_complex "overlay=x=(main_w-overlay_w)/2:y=(overlay_h)/2" -c:v libx264 -strict -2 -pix_fmt yuv420p output.mp4'

            ret = os.system(torun)
            if ret != 0:
                return Lang["CONV_FAIL"], None

            torun = 'ffmpeg -r 30 -y -i ./results/%d.png -i watermark.png -filter_complex "overlay=x=(main_w-overlay_w)/2:y=(overlay_h)/2" -c:v qtrle output_adobe_premiere.mov'
            ret = os.system(torun)

        return Lang["DONE"], "output.mp4"

with gr.Blocks(title="Live3D-v2",css="#mmd_pmx,#mmd_motion,#mmd_camera {height:70px}"  ) as ui:
    
    
    gr.Markdown("## [Live3D-v2](https://github.com/transpchan/Live3D-v2)")
    gr.Markdown(
        " [Live3D-v2](https://github.com/transpchan/Live3D-v2) (Oct. 2022) is an incremental update to the MIT-Licenced CoNR (Jul. 2022). CoNR is also known as [Live3D-v1 public beta](https://github.com/transpchan/Live3D) (Sep. 2021).  Credits:  MMD2UDP plugin by [KurisuMakise004](https://github.com/KurisuMakise004/MMD2UDP). Drawings from MIT-Licenced CoNR. ")

    with gr.Tab(Lang["VIDEO_GEN"]):
        ret = gr.Markdown(Lang["VIDEO_GEN_DESC"])

        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown(Lang["SELECT_CHAR_SHEET"])
                character_sheets = gr.File(
                    label="Character Design Sheet", file_count="multiple", file_types=[".png", ".webp"])

            with gr.Column(variant="panel"):

                # string input
                examples = gr.Text(label="UDP_Sequence", visible=False)
                gr.Markdown(Lang["SELECT_PMX"])
                mmd_pmx = gr.File(elem_id="mmd_pmx",
                    label="Zip (.zip)", file_count="single", file_types=[".zip"])

                gr.Markdown(Lang["SELECT_MOTION"])
                mmd_motion = gr.File(elem_id="mmd_motion",
                    label="MMD motion (.vmd)", file_count="single", file_types=[".vmd"])
                gr.Markdown(Lang["SELECT_CAMERA"])
                mmd_camera = gr.File(elem_id="mmd_camera",
                    label="MMD camera (.vmd)", file_count="single", file_types=[".vmd"])

                # os.system("sh download.sh")

            with gr.Column():
                run = gr.Button(Lang["RUN"])
                video = gr.Video()
                run.click(fn=conr_fn, inputs=[examples,
                          character_sheets,  mmd_pmx, mmd_motion, mmd_camera], outputs=[ret, video])

        with gr.Row():
            gr.Examples(fn=conr_fn, inputs=[examples], outputs=[ret, video],
                        examples=["short_hair", "double_ponytail"], cache_examples=False, examples_per_page=10, run_on_click=True, label=Lang["EXAMPLE"])

    with gr.Tab(Lang["UDP_POINTCLOUD"]):

        gr.Markdown(Lang["UDP_POINTCLOUD_DESC"])
        gr.Markdown(
            "## Not supported on Windows yet. Use the [Colab](https://colab.research.google.com/github/transpchan/Live3D-v2/blob/main/notebook.ipynb) instead.")
    with gr.Tab(Lang["COLORIZE_MESH"]):

        gr.Markdown(Lang["COLORIZE_MESH_DESC"])
        gr.Markdown(
            "## Not supported on Windows yet. This is usually for game/mod developers [1](https://www.bilibili.com/video/BV1ae4y1Y7ga/), [2](https://www.bilibili.com/video/BV1fB4y1V7ga), [3](https://www.youtube.com/watch?v=HK7dhP7UXzs).")

ui.launch(inbrowser=True)
