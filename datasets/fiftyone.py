import torch


def detection_to_bbox(img, det):
    bbox = det.bounding_box
    out = [
        bbox[0],
        bbox[1],
        (bbox[0] + bbox[2]),
        (bbox[1] + bbox[3]),
    ]

    return out


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, fo_ds, trans=None):
        self.fo_ds = fo_ds
        self.trans = trans
        self.clss = fo_ds.distinct("ground_truth.detections.label")
        self.file_paths = fo_ds.values("filepath")

        if self.clss[0] != "background":
            self.clss = ["background"] + self.clss
        
        self.cls_to_idx = {c: i for i, c in enumerate(self.clss)}
    
    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        x = self.fo_ds[fp]
        meta = x.metadata
        img = read_image(fp, mode=tv.io.ImageReadMode.RGB) / 255

        if self.trans:
            img = self.trans(img)

        bboxes = []
        clss = []

        for det in x.ground_truth.detections:
            bboxes.append(detection_to_bbox(img, det))
            clss.append(self.cls_to_idx[det.label])
        
        y = {
            "bboxes": torch.FloatTensor(bboxes).to(device),
            "clss": torch.LongTensor(clss).to(device),
        }
        
        return img.to(device), y
    
    def __len__(self):
        return len(self.file_paths)

