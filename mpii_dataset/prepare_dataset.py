import random 

class PrepareMpiiDataset:
    """Compute dataset for finetune."""

    def __init__(self,
                 dataset: list):
        """Initialize the dataset for finetune."""
        self.dataset = dataset    

    def opposite_gender(self, gender:str) -> str:
        if gender.__eq__("Woman"):
            return "Man"
        else:
            return "Woman"
    def opposite_race(self, race:str) -> str:
        if race.__eq__("white"):
            return random.choice(["black", "asian", "middle eastern", "latino hispanic", "indian"])
        else:
            return "white"
    def opposite_age(self, age:str) -> str:
        if age in ["child", "young"]:
            return random.choice(["middle-aged", "senior"])
        else:
            return random.choice(["child", "young"])  

    def __call__(self):
        """"""
        finetune_dataset = []
        for entry in self.dataset:
            finetune_dataset.append({
                        "anchor": entry["text"][0],
                        "negative-pair": entry["text"][0] + " " + entry["race"] + " " + entry["age"] + " " + entry["gender"],
                        "positive-pair": entry["text"][0] + " " + self.opposite_race(entry["race"]) + " " + self.opposite_age(entry["age"]) + " " + self.opposite_gender(entry["gender"]),
                        "image": entry["image"],
                        "gender": entry["gender"],
                        "race": entry["race"],
                        "age": entry["age"],
                    })
        return finetune_dataset
