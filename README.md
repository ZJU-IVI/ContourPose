# ContourPose

code for paper "ContourPose：A monocular 6D pose estimation method for reflective texture-less metal parts".

### **Video demo**

![ContourPose](figure/ContourPose.gif)

### **Pose estimation demo**

![pose demo](figure/pose_demo.png)

 

## Installation

1. Set up the python environment:

   ```
   conda create -n ContourPose python=3.7
   conda activate ContourPose
   ```

   ### install other requirements

   ```
   pip install -r requirements.txt
   ```

## Dataset Configuration
1. Prepare the dataset

   * The training and testing dataset for ContourPose can be found [here](https://github.com/ZJU-IVI/RT-Less_10parts). Unzip all files.
   
   * Download the SUN397 
   ```shell
   wget http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz
   ```
   
2. Create soft link
   ```shell
   mkdir $ROOT/data
   ln -s path/to/Real Images $ROOT/data/train
   ln -s path/to/Synthetic Images $ROOT/data/train/renders
   ln -s path/to/Synthetic Images/gtEdges $ROOT/data/train/renders/Render_edge
   ln -s path/to/Test Scenes $ROOT/data/test
   ln -s path/to/SUN2012pascalformat $ROOT/data/SUN2012pascalformat
   ```
   For more details on the file path, please refer to `dataset/Dataset.py`.

3. Object index mapping
   
   Since the dataset was still under construction when the paper was completed, the actual indexing of the objects may differ from that in the paper. Please refer to the index mapping relationships below.

| Indexing of objects in the paper (dataset) | obj1  |obj2     |   obj3  |   obj4  |  obj5   |  obj6   |   obj7  |  obj8   |   obj9  |  obj10   |
|--------------------------------------------| ----  |-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Actual indexing of objects in this code    | obj1 |  obj2   |  obj3   |  obj7   |   obj13  |   obj16  |  obj18   |  obj18   |  obj21   |   obj32  |
   
   The test scenes in which the target object is tested can be found in the `sceneObjs.yml`  file.
## Pretrained model
   Download the pretrained models from [here]() and put them to `$ROOT/model/obj{}/150.pkl` 
   
## Training and testing
1. Training
   Take the training on `obj1` as an example. 
   run  
   ```shell
   python main.py --class_type obj1 --train True
   ```

2. Testing
   Take the testing on `obj1` as an example. 

   The `sceneObjs.yml` file shows that obj1 is in scene with an index of 2.run

   ```shell
    python main.py --class_type obj1 --test True --scene 13 --index 2
   ```
