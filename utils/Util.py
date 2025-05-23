import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import BertTokenizer

from copy import deepcopy

def GetScores(boxes, label_box):

    true_box = deepcopy(label_box)

    ## Processing True_Box
    true_box *= 512

    true_box[2] += true_box[0]
    true_box[3] += true_box[1]

    true_box = true_box.unsqueeze(0)

    ious = box_iou(boxes, true_box)

    ious = ious.squeeze(1)
    return torch.argmax(ious).item()


def CreateBatchLabels(result, batch_boxes):
    indxs = []
    batch_size = len(result)
    for sample in range(batch_size):
        n = 10 if len(result[sample]) > 10 else len(result[sample])
        boxes = result[sample][:n]
        true_bbox = batch_boxes[sample]
        indxs.append(GetScores(boxes, true_bbox))

    if len(indxs) < 0:
        indxs += [0]*(10-len(indxs))

    return torch.tensor(indxs)



def plot_image(indx, batch, pred):

    plt.imshow(batch[indx].permute(1,2,0).cpu().numpy())

    ax = plt.gca()

    boxes = pred[indx]['boxes']
    labels = pred[indx]['labels']
    scores = pred[indx]['scores']

    for i in range(len(boxes)):
        if scores[i] > 0.5:
            box = boxes[i].cpu().numpy()
            label = labels[i].cpu().numpy()
            score = scores[i].cpu().numpy()

            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='r', facecolor='none'
            )

            ax.add_patch(rect)

            ax.text(
                box[0], box[1], f'{label}: {score:.2f}', color='white',
                fontsize = 8, backgroundcolor='red'
            )

    plt.show()

def plot_region_with_text(batch_imgs, batch_ids, batch_boxes, result, predict = False):
    indxs = None
    if predict == True:
        indxs = batch_boxes
    else:
        indxs = CreateBatchLabels(result, batch_boxes)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for sample, indx in enumerate(indxs):
        # decoded_str = tokenizer.decode(
        #     batch[1]['input_ids'][sample],
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=True
        # )

        # decoded_tokens = tokenizer.convert_ids_to_tokens(batch_ids[sample][0])
        # decoded_str =tokenizer.convert_tokens_to_string(decoded_tokens)

        decoded_str = tokenizer.decode(
            batch_ids[sample][0],
            skip_special_tokens=True,            # drop [CLS], [SEP], padding, etc.
            clean_up_tokenization_spaces=True    # collapse extra spaces
        )

        plt.imshow(batch_imgs[sample].permute(1,2,0).cpu().numpy())
        plt.title(decoded_str)

        ax = plt.gca()

        bbox = result[sample][indx]
        box = bbox.cpu().numpy()
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )

        ax.add_patch(rect)

        # ax.text(
        #     box[0], box[1], f'}: {score:.2f}', color='white',
        #     fontsize = 8, backgroundcolor='red'
        # )
        plt.show()
