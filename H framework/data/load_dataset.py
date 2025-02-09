from torch.utils.data import Dataset
class Feature_dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i][0],self.data[i][1],self.data[i][2],self.data[i][3],self.data[i][4],self.data[i][5]

def collate_fn(batch):
    input_seq,doc_id,label_seq,label_doc_id,label_input_id,id_label = list(zip(*batch))
    del batch
    return input_seq,doc_id,label_seq,label_doc_id,label_input_id,id_label
