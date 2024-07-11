from prediction.predict_mpii_GRA import MpiiGRAPredictor
from mpii_dataset.load_dataset import MpiiDataModule
from mpii_dataset.prepare_dataset import PrepareMpiiDataset
from numpy import save, asarray

if __name__ == "__main__":
    image_dataset = MpiiDataModule().dataset
    gra_predictor = MpiiGRAPredictor(dataset = image_dataset)
    gra_prediction = gra_predictor()
    finetune_prepare = PrepareMpiiDataset(dataset = gra_prediction)
    finetune_dataset = finetune_prepare()
    save("image_data.npy", asarray(finetune_dataset))