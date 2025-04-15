
from .utils import (
    NormalizePoseCollection,
    ReshapePoseCollection,
    ConvertToDataFrame,
)

import os
import pandas as pd
from xgboost import XGBClassifier
import copy
import trt_pose.coco
import trt_pose.models


dir_path = os.path.dirname(os.path.realpath(__file__))

class CheatDetection:
    def __init__(self):
        self.model = XGBClassifier()
        
        xgboost_model_path = os.path.join(dir_path, "XGB_BiCD_Tuned_GPU_05.model")
        self.model.load_model(xgboost_model_path)
        self.model.set_params(**{"predictor": "gpu_predictor"})  

    def DetectCheat(self, keypoints_list):
        poseCollection = []
        
        for keypoints in keypoints_list:
            pose = []
            for j,x,y in keypoints:
                if x is not None and y is not None:
                    pose.append([x, y, 1])  
                else:
                    pose.append([0, 0, 1]) 
            poseCollection.append(pose)
        cheating_keypoints = []

        if poseCollection and len(poseCollection) > 0:
            original_posecollection = copy.deepcopy(poseCollection)   
            poseCollection = NormalizePoseCollection(poseCollection)  
            poseCollection = ReshapePoseCollection(poseCollection)  
            poseCollection = ConvertToDataFrame(poseCollection)  
            
            preds = self.model.predict(poseCollection)
            
            for idx, pred in enumerate(preds):
                if pred:  
                    cheating_keypoints.append(original_posecollection[idx])  
            
            return cheating_keypoints, len(cheating_keypoints) > 0  
        
        return [], False

