from Trainer import *
from model.ModCoAttnModels import *
from utils.Dataset import *
import torch
import numpy as np
from tqdm.auto import tqdm

def main():
    train_dataloader = GetDataloader('/content/data/train', 16, 'train')
    test_dataloader = GetDataloader('/content/data/test', 16, 'test')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VisualGrounding(device)

    trainer = VisualGroundingTrainer(model, device, train_dataloader, test_dataloader)

    epochs = 40
    best_val_loss = np.inf
    best_model = None

    num_epochs_without_improvement = 0
    patience = 3

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = trainer.train_step()
        #   trainer.scheduler.step()

        test_loss, test_acc = trainer.eval_step()

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_model = model.state_dict()
            num_epochs_without_improvement = 0
        else:
            num_epochs_without_improvement += 1

        print(f'Epoch {epoch}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f} Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')
        if num_epochs_without_improvement >= patience:
            print("Early Stopping Triggered!!!")
            break
    best_model_save = VisualGrounding(device)

    # 2) Load your saved state dict
    best_model_save.load_state_dict(best_model)

    torch.save(best_model_save, "best_model.pth")

    test_dataset = VG_Dataset('/content/data/test')

    best_model_save.eval()
    best_model_save = best_model_save.to(device)

    with torch.inference_mode():

        for batch, (X_Img, X_Text, y_bbox) in enumerate(test_dataloader):
            X_Img, X_Text, y_bbox = X_Img.to(device), X_Text.to(device), y_bbox.to(device)

            roi, y_pred = best_model_save(X_Img, X_Text['input_ids'], X_Text['attention_mask'])
            y = CreateBatchLabels(roi, y_bbox).to(device)
            plot_region_with_text(X_Img, X_Text['input_ids'], y_pred.squeeze(dim=-1).argmax(dim=1), roi, predict=True)


if __name__ == '__main__':
    main()