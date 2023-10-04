import pytesseract
import os
from tqdm import tqdm
import json

# layout_dic = {"gun_control": {}, "abortion": {}}
image_dir_path = "./images/"
for dir_name in ["gun_control", "abortion"]:
    dir_path = image_dir_path + dir_name
    # dir_dic = {}
    for path in tqdm(os.listdir(dir_path)):
        tweet_id = path.split('.')[0]

        out_path = "./hocr/" + dir_name + "/" + tweet_id + ".html"
        if not os.path.exists(out_path):
            image_path = dir_path + '/' + path
            hocr = pytesseract.image_to_pdf_or_hocr(image=image_path, extension='hocr')
            hocr = hocr.decode('utf-8')

            with open(out_path, "w", encoding='utf-8') as f:
                f.write(hocr)
    #     string_list = hocr.split('\n')
    #     bboxes = []
    #     tokens = []
    #     for line in string_list:
    #         if "<span class='ocrx_word'" in line:
    #             bbox_start = line.find('bbox') + 5
    #             bbox_end = line.find('; x_wconf')
    #             bbox_string = line[bbox_start:bbox_end]
    #             bbox = [int(num) for num in bbox_string.split(' ')]
    #             bboxes.append(bbox)
    #             token_start = line.find('\'>') + 2
    #             token_end = line.find('</span>')
    #             token = line[token_start:token_end]
    #             tokens.append(token)
    #     dir_dic[tweet_id] = {"tokens": tokens, "bboxes": bboxes}
    # layout_dic[dir_name] = dir_dic

# out_path = "./layout_dic.json"
# json_str = json.dumps(layout_dic, indent=4)
# with open(out_path, "w") as f:
#     f.write(json_str)
