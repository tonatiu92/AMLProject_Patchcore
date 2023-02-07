import torch
import utils
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageFilter
import scipy.ndimage as ndimage

class Patchcore:
    
    def __init__(self,autoencoder,sampler,resize:int = 256, imagesize:int = 224, patchsize:int = 3, patchstride:int=1, dilation:int=1,pretrain_embed:int = 1024, target_embed = 1024, device = None, nb_neighbours:int = 3)->None:
        self._autoencoder = autoencoder
        self._sampler = sampler
        self._resize = resize
        self._imagesize =imagesize
        self._patchsize = patchsize
        self._patchstride = patchstride
        self._dilation = dilation
        self._pretrain_embed = pretrain_embed
        self._target_embed = target_embed
        self.device = device
        self.memory_bank = None
        self.nb_neighbours = nb_neighbours
        
    def fit(self, dataloader) -> None:
        """
        Description: fit the memory_bank, defining them and concatening them
        
        @params:
        datasets [torch.Tensor]: The dataset to be loaf
        """
        imagesize = dataloader.dataset._imagesize
        features_list = []
        for image in dataloader:
            image = image['image']
            if self.device:
                #Part 1: Execute resnet with features extraction as recomended in the article jextracted[2,3]
                with torch.no_grad():
                    image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features = self._autoencoder(image.to(self.device)) #apply the forward  function
                if type(features) is dict:
                    """
                    For clip features
                    """
                    features = [features[layer] for layer in features ]
                features_list.append(self.local_patch_features(features, self.device)[0])
        patch = torch.cat(features_list, axis=0) ### Group features
        self.memory_bank = self._sampler.run(patch,self._pretrain_embed)#apply the coreset to select inside the memory bank
    
    def predict(self, dataloader)->list:
        """
        Description: predict according the memory bank
        """
        scores = []
        for image in dataloader:
            label = image["anomaly"]
            image = image["image"]
            image = image.to(torch.float).to(self.device)
            #embed part
            with torch.no_grad():
                #get the memory_bank from the test images
                with torch.no_grad():
                    features_map = self._autoencoder(image.to(self.device)) #apply the forward resnet function
                if type(features_map) is dict:
                    """
                    For clip features
                    """
                    features_map = [features_map[layer] for layer in features_map ]
                features, nb_patch = self.local_patch_features(features_map, self.device)
                
                neareset_neighbours_index, new_feature, maximum_score_value, min_distance = self.maximum_distance_score(features, self.memory_bank)

                final_score = self.increasing_score(neareset_neighbours_index,new_feature, features,maximum_score_value)
                
                #realigning computed patch anomaly scores based on their respective spatial location
                _seg = min_distance.view(1,1,*features_map[0].shape[-2:])
                #interpolation of the final score map to get the size of original image
                #image_score_map = utils.bilinear_interpolate(#   image_score_map, output_size=(self._imagesize,self._imagesize)#)In this case we cannot use our own method because it is too long
                _seg = F.interpolate(_seg,size=(self._imagesize,self._imagesize),mode='bilinear', align_corners=False)
                _seg = _seg.to("cpu")
                image_map = self.segmentation(_seg)
                
                scores.append({"score":(final_score,label), "mask": [mask for mask in image_map]})
        return scores
     
    def local_patch_features(self,image: torch.Tensor, device) -> tuple:
        """
        Description: extract the memory_bank for a selectded image
        @args:
        image [torch.Tensor]: the selected image
        device: the selected gpu
        
        @returns
        tuple (features of the image, shape of the patch)
        """
            
        # Part2: Extract memory_bank
        #Now we get a tensor for each layer extraction define in the ouput indices
        #For each tensor we will try to get the different memory_bank
        features = image
        features = [self.patchify(x,self._patchsize,self._patchstride) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]

        #PART 3 Re-Scaling the features to improve resolution and match both patch features collections
        features[1] = self.scaling_to_low_level(features[1], patch_shapes)
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        #Preprocessing (1) or (3) in the Article for each feature
        feat = []
        for feature in features:
            feature = feature.to(device)
            feature = feature.reshape(len(feature), 1, -1)
            feature = F.adaptive_avg_pool1d(feature, self._pretrain_embed).squeeze(1)
            feat.append(feature)
        features = torch.stack(feat,dim=1)

        #Aggregator (2) in the Article
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self._target_embed)
        return features.reshape(len(feature),-1), patch_shapes  
      
    def scaling_to_low_level(self,input: torch.Tensor, patch_shapes: list, ) -> torch.Tensor:
        """
        Description
            - It corresponds to the last paragraph of section 3.1 
            - Patchcore uses only two intermediate features (here [2,3])
            - aggregating each element with the lowest hierarchy level
            - This part scale all the features on the reference one 
            start shape: batch_size, nb_blocks of features[i], channels, patchsize, patchsize
            end shape: batch_size, nb_blocks of ref features, channels, patchsize, patchsize
        @args
        input [torch.tensor]: the input image
        patch_shape: the shape of the patch
        
        @tensor
        return the tensor with the new resolution
        """
        _features = input
        patch_dims = patch_shapes[1] #The initial_dim_of_ the intermediate
        batch_size = _features.shape[0]
        channels = _features.shape[2]
        patchsize = _features.shape[-1]
        
        main_shape = [batch_size, channels, patchsize, patchsize]
        
        #we remove the block dimension batchsize, initial height, initial width, channels, patchsize, patchsize
        _features = _features.reshape(
            _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
        )

        #reorder the tensor: batchsize, channels, patchsize, patchsize, initial height, initial width
        _features = _features.permute(0, -3, -2, -1, 1, 2)
        
        #reshape: batchsize*channels*patchsize*patchsize, initial height, initial width
        _features = _features.reshape(-1, *_features.shape[-2:])

        #bilinear interpolation to improve the resolution (add dimension for the tensor in order to get 4-D tensor)
        _features = utils.bilinear_interpolate(_features.unsqueeze(1), (patch_shapes[0][0], patch_shapes[0][1])) #output to lowest hierarchy level feature

        #remove the dim 1 added during interpolation
        _features = _features.squeeze(1)
        
        _features = _features.reshape(
            *main_shape[:], patch_shapes[0][0], patch_shapes[0][1]
        )

        _features = _features.permute(0, -2, -1, 1, 2, 3)

        #final reshape to get the same shape format as the ref one
        _features = _features.reshape(len(_features), -1, *_features.shape[-3:])

        return _features
    
    def patchify(self,features, patchsize: int, stride:int, dilation:int = 1)->tuple:
        """
        Description: This function allow us to get the differents memory_bank provided of the features

        @params:
        features [torch.Tensor]
        patchsize [int]
        stride: [int]
        dilation [int]

        @returns:
        a tensor of size (batch_size, nb_blocks, nb_channels, kernel_size_height, kernel_size_width)
        nb_memory_bank[int, int]:  it is the dimension of the total matrix filtered taking in account padding, dilation and stride.

        """
        padding = (patchsize-1)//2  #define the padding to fill the border

        #Perform an unfold on the features
        unfolder = torch.nn.Unfold(kernel_size=patchsize, stride=stride, padding=padding, dilation=dilation)
        unfolded_features = unfolder(features)

        #After Unfold we get a feature of Size(batch_size, nb_channels*kernel_size**2, nbBlocks)
        nb_memory_bank = [None, None]

        #For each dimension (Height and Width) get the number of blocks
        nb_memory_bank[0] = ((features.shape[-2] + 2 * padding - dilation * (patchsize - 1) - 1) // stride) + 1 #out_height
        nb_memory_bank[1] = ((features.shape[-1] + 2 * padding - dilation * (patchsize - 1) - 1) // stride) + 1#out_width out_heigt*out_width = nbBlocks
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], patchsize, patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        #After reshape we get a tensor of Size (batch_size, nb_blocks, nb_channels, kernel_size_height, kernel_size_width)
        return unfolded_features,nb_memory_bank
  
    def maximum_distance_score(self, features: torch.Tensor, patch_memory_bank: torch.Tensor)->tuple:
        """
        Description: Computing equation 6 of the paper. We want to calculate the maximum distance score
        
        @args:
        feature [torch.Tensor]: the test feature
        patch_memory_bank [torch.Tensor]: the memory bank feature
        
        @returns
        eareset_neighbours_index, new_feature, maximum_score_value, min_distance

        
        """
        with torch.no_grad():
            #find the distances between memory_bank and test set
            distances = torch.cdist(features, patch_memory_bank)
            
            #getting the min value and its index from distances 
            min_distance, index_min_distance = torch.min(distances, dim=1)
            
            #Now inside the patch closest to the feature take max value and index
            maximum_score_index = torch.argmax(min_distance)
            maximum_score_value = torch.max(min_distance)
            
            #Looking at the anomalous patch taking in the test feature and memory bank feature
            new_feature = features[maximum_score_index].unsqueeze(0)
            memory_bank_feature = patch_memory_bank[index_min_distance[maximum_score_index]].unsqueeze(0)
            
            w_dist = torch.cdist(memory_bank_feature, patch_memory_bank) # find knn to m_star pt.1
            
            #index of the topk n neighbours
            _, neareset_neighbours_index = torch.topk(w_dist, k=self.nb_neighbours, largest=False) 
            
            return neareset_neighbours_index, new_feature, maximum_score_value, min_distance

    def increasing_score(self, nn_index: list, new_feature: torch.Tensor, features: torch.Tensor, maximum_score_value):
        """
        Description: Equation 7 of the paper
        
        @args
        nn_index [list]: list of the nearest neigbours index
        new_feature [torch.Tensor]: The m_test of article
        features [torch.Tensor]: the patch of the test image
        maximum_score_value: the s* from the article
        
        @returns:
        
        the reweighted_score 
        
        """
        # equation 7 from the paper
        with torch.no_grad():
            m_star_knn = torch.linalg.norm(new_feature-self.memory_bank[nn_index[0,1:]], dim=1)
            D = torch.sqrt(torch.tensor(features.shape[1]))
            weight = 1-(torch.exp(maximum_score_value/D)/(torch.sum(torch.exp(m_star_knn/D))))
            score = weight*maximum_score_value
        
        return score
    
    def segmentation(self, image):
        """
        Description: Segmentation function
        
        @args:
        image: input image map
        """
        loader = transforms.ToTensor()
        unloader = transforms.ToPILImage()
        image = image.to(self.device)
        maximum = image.max()
        #As in the article we apply a gaussian of 4
        loaded = loader(unloader(image[0]/maximum).filter(ImageFilter.GaussianBlur(radius = 4))).to(self.device)*maximum
        return loaded