import gradio as gr
import requests
import json
from PIL import Image

api_endpoint = {
    "unet": 'http://127.0.0.1:5110/generate',
    "snunet": 'http://127.0.0.1:5111/generate',
    "fc_ef_diff": 'http://127.0.0.1:5112/generate',
    "fc_ef_conc": "http://127.0.0.1:5113/generate",
    "changeformer": "http://127.0.0.1:5114/generate",
    "adhr_cdnet": "http://127.0.0.1:5115/generate",
    "bam_cd": "http://127.0.0.1:5116/generate"
}

def generate_mask(model, before_img_f, after_img_f, label_f=None):
    req_body = {
        "before_img_fp":before_img_f.name,
        "after_img_fp":after_img_f.name,
    }
    if label_f is not None:
        req_body["label_fp"] = label_f.name
    
    response = requests.post(api_endpoint[model], json=req_body)
    json_res = json.loads(response.text)
    return [
        (Image.open(json_res["before_img_fp"]), "S-2 image before the wildfire"), 
        (Image.open(json_res["after_img_fp"]), "S-2 image after the wildfire"), 
        (Image.new(mode='RGB', size=(256,256), color=(0,0,0)) if label_f is None else Image.open(json_res["label_img_fp"]), "Ground truth image segmentations of the burnt area"), 
        (Image.open(json_res["predictions_img_fp"]), "Model prediction image segmentation of the burnt area")
        ]
demo_ui = gr.Interface(
    inputs = [
        gr.Dropdown(list(api_endpoint.keys()), value=list(api_endpoint.keys())[0], label='Select the model to predict', show_label=True),
        gr.File(
            file_types=['.npy'],
            label="Before Image. Upload the .npy file containing image data of the area before the wildfire.",
            show_label=True
        ),
        gr.File(
            file_types=['.npy'],
            label="After Image. Upload the .npy file containing image data of the area after the wildfire.",
            show_label=True
        ),
        gr.File(
            file_types=['.npy'],
            label="Label. If available, upload the .npy file containing the manually segmented burnt area.",
            show_label=True
        )
    ],
    fn=generate_mask,
    outputs=[
        gr.Gallery(format='png', columns=2, rows=2)
    ],
    examples=[
        [list(api_endpoint.keys())[0],
         './data/dataset/2021/sample00000020_145_2021.sen2_60_pre.npy',
         './data/dataset/2021/sample00000020_145_2021.sen2_60_post.npy',
         './data/dataset/2021/sample00000020_145_2021.label.npy'
         ],
         [list(api_endpoint.keys())[0],
          './data/dataset/2021/sample00000004_75_2021.sen2_60_pre.npy',
          './data/dataset/2021/sample00000004_75_2021.sen2_60_post.npy',
          './data/dataset/2021/sample00000004_75_2021.label.npy'
          ],
          [list(api_endpoint.keys())[0],
           './data/dataset/2021/sample00000011_64_2021.sen2_60_pre.npy',
          './data/dataset/2021/sample00000011_64_2021.sen2_60_post.npy',
          './data/dataset/2021/sample00000011_64_2021.label.npy'
          ],
          [list(api_endpoint.keys())[0],
           './data/dataset/2020/sample00000020_46_2020.sen2_60_pre.npy',
          './data/dataset/2020/sample00000020_46_2020.sen2_60_post.npy',
          './data/dataset/2020/sample00000020_46_2020.label.npy'
          ],
          [list(api_endpoint.keys())[0],
            './data/dataset/2021/sample00000020_64_2021.sen2_60_pre.npy',
          './data/dataset/2021/sample00000020_64_2021.sen2_60_post.npy',
          './data/dataset/2021/sample00000020_64_2021.label.npy'
          ],
          [list(api_endpoint.keys())[0],
            './data/dataset/2021/sample00000037_138_2021.sen2_60_pre.npy',
          './data/dataset/2021/sample00000037_138_2021.sen2_60_post.npy',
          './data/dataset/2021/sample00000037_138_2021.label.npy'
          ],
          [list(api_endpoint.keys())[0],
            './data/dataset/2021/sample00000038_138_2021.sen2_60_pre.npy',
          './data/dataset/2021/sample00000038_138_2021.sen2_60_post.npy',
          './data/dataset/2021/sample00000038_138_2021.label.npy'
          ],
          [list(api_endpoint.keys())[0],
            './data/dataset/2018/sample00000052_10_2018.sen2_60_pre.npy',
          './data/dataset/2018/sample00000052_10_2018.sen2_60_post.npy',
          './data/dataset/2018/sample00000052_10_2018.label.npy'
          ],
          [list(api_endpoint.keys())[0],
            './data/dataset/2021/sample00000052_62_2021.sen2_60_pre.npy',
          './data/dataset/2021/sample00000052_62_2021.sen2_60_post.npy',
          './data/dataset/2021/sample00000052_62_2021.label.npy'
          ],
          [list(api_endpoint.keys())[0],
            './data/dataset/2021/sample00000061_62_2021.sen2_60_pre.npy',
          './data/dataset/2021/sample00000061_62_2021.sen2_60_post.npy',
          './data/dataset/2021/sample00000061_62_2021.label.npy'
          ],
    ]
)

demo_ui.launch(root_path="/burnt-area-mapping/", server_name='0.0.0.0', share=True)
