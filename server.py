import timm
import torch
import logging
import albumentations as A

from skimage.io import imread
from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

app = Flask(__name__)

model = timm.create_model('tiny_vit_5m_224.dist_in22k', num_classes=44, pretrained=False)

model.eval()

state = torch.load('model.pth')
model.load_state_dict(state)

ages = [ idx for idx in range(3, 27, 3) ] + [ 24, 28, 32, 36 ] + [ idx for idx in range(42, 19*12+6, 6)]

c2a = {c:a for c, a in enumerate(ages)}


@app.route("/ready")
def ready():
    return "OK"

@app.route("/predict", methods=["POST"])
def recognize():
    imagefile = request.files.get('file', '')

    img = imread(imagefile)

    eval_pipe = A.Sequential([
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(min_height=512, min_width=512),
    ])

    img = eval_pipe(image=img)['image']

    if len(img.shape) == 2:
        img = torch.tensor(img)[None]
        img = img.repeat(3, 1, 1).float()

    else:
        img = torch.tensor(img).float()
        img = img.permute(2, 1, 0)


    with torch.no_grad():
        class_ = int(model(img[None])[0].argmax(0).numpy())

    boneage = c2a[class_]

    data = {'boneage': boneage}
    return jsonify(data)

#     os.makedirs(os.path.join(results_dir, str(task_id)), exist_ok=True)
#     request_data = request.files["file"]
#     logger.debug("IN RECOGNIZE")
#     is_valid_request(request_data, logger)

#     datapath = os.path.join(uploads_dir, request_data.filename)
#     request_data.save(datapath)
#     response = model_inference(datapath, task_id)

#     return response  # message with link


# @app.route("/results/<path:filename>", methods=["GET", "POST"])
# def download(filename):
#     task_id, filename = filename.split("/")
#     results_dir_task = os.path.join(results_dir, str(task_id))

#     logger.info(f"DOWNLOADING task {task_id} from {results_dir}")
#     return send_from_directory(results_dir_task, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010, debug=True)
