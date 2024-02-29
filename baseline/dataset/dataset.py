import torch
import torch.utils.data as data
import pickle
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms

CACHE_LOCATION = '/home/stud-1/aditya/baseline/dataset/cache'

# utils
def sliding_window_split(df, seq_len):
    windowed_dfs = []
    
    for end_idx in range(len(df)):
        # Calculate the start index for slicing the dataframe. Ensure it's not negative.
        start_idx = max(0, end_idx - seq_len + 1)
        # Slice the dataframe from start_idx to end_idx (inclusive)
        window_df = df.iloc[start_idx:end_idx+1].reset_index(drop=True)
        windowed_dfs.append(window_df)
        
    return windowed_dfs

class DatasetEXPR(data.Dataset):
    def __init__(self, annotation_file, mode='Train', seq_len=32, transform=None):
        assert (mode == 'Train' or mode == 'Validation'), "mode should be either 'Train' or 'Validation'"
        assert os.path.exists(annotation_file), "Annotation file path doesn't exist"
        

        # open dataset
        with open(annotation_file, 'rb') as f:
            anot_dict = pickle.load(f)
        if mode == 'Train':
            _data = anot_dict['EXPR_Set']['Train_Set']
        else :
            _data = anot_dict['EXPR_Set']['Validation_Set']
        
        # divide each video into segments of seq_len
        data = []
        print('preparing dataset for EXPR')
        ## check if cache exists
        if os.path.exists(f'{CACHE_LOCATION}/data_splitted.EXPR'):
            with open(f'{CACHE_LOCATION}/data_splitted.EXPR', 'rb') as f:
                data = pickle.load(f)
        else :
            for _video in tqdm(_data):
                for _window in sliding_window_split(_data[_video], seq_len):
                    data.append(_window.to_dict())
            print("standardizing dataset")
            for i in tqdm(range(len(data))):
                if len(data[i]['label']) < 32:
                    _i = len(data[i]['label'])
                    while(1):
                        if(_i == 32):
                            break 
                        data[i]['label'][_i] = -1
                        data[i]['path'][_i] = 'blank'
                        data[i]['frames_ids'][_i] = -1
                        _i += 1

            with open(f'{CACHE_LOCATION}/data_splitted.EXPR', 'wb') as f:
                pickle.dump(data, f)

        self.data = data
        self.transform = transform
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, index): 
        assert (index < len(self.data))
        item_data = self.data[index]
        # type of the data will be 
        # item_data = {
        #     'label' : {},
        #     'path' : {},
        #     'frame_ids' : {}
        # }

        blank_images = []
        blank_paths = []
        blank_frame_ids = []
        blank_labels = []

        non_blank_images = []
        non_blank_paths = []
        non_blank_frame_ids = []
        non_blank_labels = []

        for path, label, frame_id in zip(item_data['path'].values(), item_data['label'].values(), item_data['frames_ids'].values()):
            if path == 'blank':
                image = Image.new('RGB', (256, 256), color=(0, 0, 0))
                blank_images.append(image)
                blank_paths.append(path)
                blank_frame_ids.append(frame_id)
                blank_labels.append(label)
            else:
                image = Image.open(path).convert('RGB')
                non_blank_images.append(image)
                non_blank_paths.append(path)
                non_blank_frame_ids.append(frame_id)
                non_blank_labels.append(label)

        images = blank_images + non_blank_images
        paths = blank_paths + non_blank_paths
        frame_ids = blank_frame_ids + non_blank_frame_ids
        labels = blank_labels + non_blank_labels
        
        labels = [torch.tensor(label) for label in labels]
        labels_tensor = torch.stack(labels)
        frame_ids = [torch.tensor(frame_id) for frame_id in frame_ids]
        frame_ids_tensor = torch.stack(frame_ids)
        if self.transform is None :
            to_tensor = transforms.ToTensor()
            images_tensor = torch.stack([to_tensor(image) for image in images])
        else :
            to_tensor = self.transform
            images_tensor = torch.stack([to_tensor(image) for image in images])
        
        _data = {
            'labels': labels_tensor,
            'paths': paths,
            'frame_ids': frame_ids_tensor,
            'images': images_tensor
        }
        
        return _data
        
class DatasetAU(data.Dataset):
    def __init__(self, annotation_file, mode='Train', seq_len=32, transform=None):
        assert (mode == 'Train' or mode == 'Validation'), "mode should be either 'Train' or 'Validation'"
        assert os.path.exists(annotation_file), "Annotation file path doesn't exist"
        

        # open dataset
        with open(annotation_file, 'rb') as f:
            anot_dict = pickle.load(f)
        if mode == 'Train':
            _data = anot_dict['AU_Set']['Train_Set']
        else :
            _data = anot_dict['AU_Set']['Validation_Set']

        # divide each video into segments of seq_len
        data = []
        print('preparing dataset for AU')
        ## check if cache exists
        if os.path.exists(f'{CACHE_LOCATION}/data_splitted.AU'):
            with open(f'{CACHE_LOCATION}/data_splitted.AU', 'rb') as f:
                data = pickle.load(f)
        else :
            for _video in tqdm(_data):
                for _window in sliding_window_split(_data[_video], seq_len):
                    data.append(_window.to_dict())
            print("standardizing dataset")
            for i in tqdm(range(len(data))):
                if len(data[i]['AU1']) < 32:
                    _i = len(data[i]['AU1'])
                    while(1):
                        if(_i == 32):
                            break 
                        data[i]['AU1'][_i] = -1
                        data[i]['AU2'][_i] = -1
                        data[i]['AU4'][_i] = -1
                        data[i]['AU6'][_i] = -1
                        data[i]['AU12'][_i] = -1
                        data[i]['AU15'][_i] = -1
                        data[i]['AU20'][_i] = -1
                        data[i]['AU25'][_i] = -1
                        data[i]['path'][_i] = 'blank'
                        data[i]['frames_ids'][_i] = -1
                        _i += 1

            with open(f'{CACHE_LOCATION}/data_splitted.AU', 'wb') as f:
                pickle.dump(data, f)

        self.data = data
        self.transform = transform
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, index): 
        assert (index < len(self.data))
        item_data = self.data[index]
        # type of the data will be 
        # item_data = {
        #     'AU1' : {},
        #     'AU2' : {},
        #     'AU4' : {},
        #     'AU6' : {},
        #     'AU12' : {},
        #     'AU15' : {},
        #     'AU20' : {},
        #     'AU25' : {},
        #     'path' : {},
        #     'frame_ids' : {}
        # }

        images = []
        paths = []
        frame_ids = []
        AUs = []

        blank_images = []
        blank_paths = []
        blank_frame_ids = []
        blank_AUs = []

        non_blank_images = []
        non_blank_paths = []
        non_blank_frame_ids = []
        non_blank_AUs = []

        for index, (path, frame_id) in enumerate(zip(item_data['path'].values(), item_data['frames_ids'].values())):
            _AUs = []
            for au_key in ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25']: 
                _AUs.append(item_data[au_key][index]) 
            _AUs_tensor = torch.stack([torch.tensor(au, dtype=torch.float32) for au in _AUs])
            
            if path == 'blank':
                blank_images.append(Image.new('RGB', (256, 256), color=(0, 0, 0)))
                blank_paths.append(path)
                blank_frame_ids.append(frame_id)
                blank_AUs.append(_AUs_tensor)
            else:
                image = Image.open(path).convert('RGB')
                non_blank_images.append(image)
                non_blank_paths.append(path)
                non_blank_frame_ids.append(frame_id)
                non_blank_AUs.append(_AUs_tensor)

        images = blank_images + non_blank_images
        paths = blank_paths + non_blank_paths
        frame_ids = blank_frame_ids + non_blank_frame_ids
        AUs = blank_AUs + non_blank_AUs
        
        AUs_tensor = torch.stack(AUs)
        frame_ids = [torch.tensor(frame_id) for frame_id in frame_ids]
        frame_ids_tensor = torch.stack(frame_ids)
        if self.transform is None :
            images_tensor = torch.stack(images)
        else :
            to_tensor = self.transform
            images_tensor = torch.stack([to_tensor(image) for image in images])
        
        _data = {
            'AUs': AUs_tensor,
            'paths': paths,
            'frame_ids': frame_ids_tensor,
            'images': images_tensor
        }
        
        return _data

class DatasetVA(data.Dataset):
    def __init__(self, annotation_file, mode='Train', seq_len=32, transform=None):
        assert (mode == 'Train' or mode == 'Validation'), "mode should be either 'Train' or 'Validation'"
        assert os.path.exists(annotation_file), "Annotation file path doesn't exist"
        

        # open dataset
        with open(annotation_file, 'rb') as f:
            anot_dict = pickle.load(f)
        if mode == 'Train':
            _data = anot_dict['VA_Set']['Train_Set']
        else :
            _data = anot_dict['VA_Set']['Validation_Set']

        # divide each video into segments of seq_len
        data = []
        print('preparing dataset for VA')
        ## check if cache exists
        if os.path.exists(f'{CACHE_LOCATION}/data_splitted.VA'):
            with open(f'{CACHE_LOCATION}/data_splitted.VA', 'rb') as f:
                data = pickle.load(f)
        else :
            for _video in tqdm(_data):
                for _window in sliding_window_split(_data[_video], seq_len):
                    data.append(_window.to_dict())
            print("standardizing dataset")
            for i in tqdm(range(len(data))):
                if len(data[i]['valence']) < 32:
                    _i = len(data[i]['valence'])
                    while(1):
                        if(_i == 32):
                            break 
                        data[i]['valence'][_i] = 0
                        data[i]['arousal'][_i] = 0
                        data[i]['path'][_i] = 'blank'
                        data[i]['frames_ids'][_i] = -1
                        _i += 1

            with open(f'{CACHE_LOCATION}/data_splitted.VA', 'wb') as f:
                pickle.dump(data, f)

        self.data = data
        self.transform = transform
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, index): 
        assert (index < len(self.data))
        item_data = self.data[index]
        # type of the data will be 
        # item_data = {
        #     'valence' : {},
        #     'arousal' : {},
        #     'path' : {},
        #     'frame_ids' : {}
        # }

        images = []
        paths = []
        valences = []
        arousals = []
        frame_ids = []

        blank_data = []
        non_blank_data = []

        for path, valence, arousal, frame_id in zip(item_data['path'].values(), item_data['valence'].values(), item_data['arousal'].values(), item_data['frames_ids'].values()):
            if path == 'blank':
                image = Image.new('RGB', (256, 256), color=(0, 0, 0)) 
                blank_data.append((image, path, valence, arousal, frame_id))
            else:
                image = Image.open(path).convert('RGB') 
                non_blank_data.append((image, path, valence, arousal, frame_id))
        
        combined_data = blank_data + non_blank_data

        for image, path, valence, arousal, frame_id in combined_data:
            images.append(image)
            paths.append(path)
            valences.append(valence)
            arousals.append(arousal)
            frame_ids.append(frame_id)
        
        valences = [torch.tensor(valence) for valence in valences]
        arousals = [torch.tensor(arousal) for arousal in arousals]
        valences_tensor = torch.stack(valences)
        arousals_tensor = torch.stack(arousals)
        frame_ids = [torch.tensor(frame_id) for frame_id in frame_ids]
        frame_ids_tensor = torch.stack(frame_ids)
        if self.transform is None :
            images_tensor = torch.stack(images)
        else :
            to_tensor = self.transform
            images_tensor = torch.stack([to_tensor(image) for image in images])
        
        _data = {
            'valence': valences_tensor,
            'arousal': arousals_tensor,
            'paths': paths,
            'frame_ids': frame_ids_tensor,
            'images': images_tensor
        }
        
        return _data

if __name__ == "__main__":
    print("inside main")
    annotation_file = '/home/stud-1/aditya/datasets/affwild2/Third ABAW Annotations/annotations.pkl'
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # dataset = DatasetVA(annotation_file=annotation_file, mode='Train', seq_len=32, transform=transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    # dataset = DatasetAU(annotation_file=annotation_file, mode='Train', seq_len=32, transform=transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    dataset = DatasetEXPR(annotation_file=annotation_file, mode='Train', seq_len=32, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}")
        # print(f"Image Tensor Shape: {batch['images'].shape}")
        # print(f"Labels: {batch['valence']}")
        # print(f"Labels: {batch['arousal']}")
        print(f"Labels: {batch['labels']}")
        # print(f"Labels: {batch['AUs']}")
        # print(f"Frame IDs: {batch['frame_ids']}")
        print(f"paths : {batch['paths']}")
        # Break after the first batch to keep the output short
        # break