
(Accident_Detection_and_Alert_System_Using_Yolov11+GAN) D:\Accident_Detection_and_Alert_System_Using_Yolov11+GAN>yolo detect train data=data.yaml model=yolo11n.pt epochs
=50 imgsz=640 device=0 
Ultralytics 8.3.77 🚀 Python-3.12.3 torch-2.6.0+cu126 CUDA:0 (NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144MiB)
engine\trainer: task=detect, mode=train, model=yolo11n.pt, data=data.yaml, epochs=50, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=8, project=None, name=train10, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs\detect\train10
Overriding model.yaml nc=80 with nc=10

                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]
 23        [16, 19, 22]  1    432622  ultralytics.nn.modules.head.Detect           [10, [64, 128, 256]]
YOLO11n summary: 181 layers, 2,591,790 parameters, 2,591,774 gradients, 6.5 GFLOPs

Transferred 448/499 items from pretrained weights
TensorBoard: Start with 'tensorboard --logdir runs\detect\train10', view at http://localhost:6006/
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning D:\Accident_Detection_and_Alert_System_Using_Yolov11+GAN\datasets\train\labels.cache... 5849 images, 0 backgrounds, 0 corrupt: 100%|██████████| 5849/584 
val: Scanning D:\Accident_Detection_and_Alert_System_Using_Yolov11+GAN\datasets\valid\labels.cache... 712 images, 0 backgrounds, 0 corrupt: 100%|██████████| 712/712 [00
Plotting labels to runs\detect\train10\labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.000714, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
TensorBoard: model graph visualization added ✅
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs\detect\train10
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      2.39G      1.167      2.333      1.141         77        640: 100%|██████████| 366/366 [01:26<00:00,  4.21it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.43it/s]
                   all        712       3335      0.598      0.411      0.429      0.302

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      2.54G      1.056      1.332      1.091         66        640: 100%|██████████| 366/366 [01:19<00:00,  4.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.34it/s]
                   all        712       3335      0.653      0.462      0.513      0.366

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      2.33G      1.014      1.177      1.075         42        640: 100%|██████████| 366/366 [01:18<00:00,  4.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.50it/s]
                   all        712       3335      0.656      0.499       0.49      0.355

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50      2.34G      1.002      1.073      1.073         48        640: 100%|██████████| 366/366 [01:19<00:00,  4.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.58it/s]
                   all        712       3335      0.625      0.558      0.569      0.422

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50      2.37G     0.9501     0.9644      1.049         66        640: 100%|██████████| 366/366 [01:19<00:00,  4.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.78it/s]
                   all        712       3335      0.696      0.585      0.602      0.466

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50      2.32G     0.9306     0.9012      1.037         77        640: 100%|██████████| 366/366 [01:19<00:00,  4.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.62it/s]
                   all        712       3335      0.703      0.579      0.639      0.502

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50      2.45G     0.9129     0.8517      1.028         36        640: 100%|██████████| 366/366 [01:19<00:00,  4.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.58it/s]
                   all        712       3335      0.805      0.621      0.714      0.566

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50      2.48G     0.8867     0.8166       1.02         71        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.64it/s]
                   all        712       3335      0.765      0.672       0.71      0.555

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50      2.37G     0.8798      0.788      1.012         70        640: 100%|██████████| 366/366 [01:19<00:00,  4.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.17it/s]
                   all        712       3335      0.848      0.675      0.753      0.588

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50      2.46G     0.8738     0.7645      1.012         53        640: 100%|██████████| 366/366 [01:20<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.63it/s]
                   all        712       3335      0.836      0.647       0.75      0.604

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50      2.39G     0.8605     0.7466      1.008         46        640: 100%|██████████| 366/366 [01:20<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.30it/s]
                   all        712       3335      0.833      0.697      0.784      0.631

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50      2.36G     0.8616     0.7275      1.004         32        640: 100%|██████████| 366/366 [01:20<00:00,  4.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.35it/s]
                   all        712       3335      0.852      0.692      0.789      0.638

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50      2.44G     0.8351     0.7079     0.9942         67        640: 100%|██████████| 366/366 [01:19<00:00,  4.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.72it/s]
                   all        712       3335      0.854      0.701       0.83      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50      2.47G     0.8284     0.6888     0.9914         56        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.36it/s]
                   all        712       3335      0.856      0.708      0.785      0.631

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50      2.42G     0.8243     0.6749     0.9913         70        640: 100%|██████████| 366/366 [01:19<00:00,  4.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.66it/s]
                   all        712       3335      0.824      0.756      0.816      0.666

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50      2.44G     0.8155     0.6587     0.9892         56        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.32it/s]
                   all        712       3335      0.865      0.714      0.846      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50      2.45G     0.8193     0.6537     0.9857         48        640: 100%|██████████| 366/366 [01:20<00:00,  4.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.65it/s]
                   all        712       3335       0.91      0.725      0.887      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/50      2.47G     0.7989     0.6406     0.9811         68        640: 100%|██████████| 366/366 [01:20<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.63it/s]
                   all        712       3335      0.879      0.777      0.897      0.748

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/50      2.43G     0.7919     0.6284     0.9764         70        640: 100%|██████████| 366/366 [01:19<00:00,  4.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.74it/s]
                   all        712       3335      0.925      0.764      0.917      0.763

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/50      2.36G     0.7949     0.6173     0.9776         71        640: 100%|██████████| 366/366 [01:20<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.57it/s]
                   all        712       3335      0.865      0.758      0.878      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/50      2.52G     0.7883     0.6074     0.9741         56        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.58it/s]
                   all        712       3335      0.868      0.785      0.879      0.734

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/50      2.37G     0.7759     0.6024     0.9714         90        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.66it/s]
                   all        712       3335      0.864      0.797      0.914      0.775

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/50      2.44G     0.7756       0.59     0.9685         62        640: 100%|██████████| 366/366 [01:20<00:00,  4.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:06<00:00,  3.79it/s]
                   all        712       3335      0.899      0.779      0.939      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/50      2.46G     0.7721     0.5745     0.9672         66        640: 100%|██████████| 366/366 [01:19<00:00,  4.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.63it/s]
                   all        712       3335      0.915      0.771      0.917      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/50      2.41G     0.7623     0.5714     0.9653         60        640: 100%|██████████| 366/366 [01:20<00:00,  4.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.60it/s]
                   all        712       3335      0.936      0.787      0.922      0.773

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/50      2.43G      0.756     0.5654     0.9627         72        640: 100%|██████████| 366/366 [01:20<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.65it/s]
                   all        712       3335      0.735      0.915      0.911      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/50      2.37G     0.7578     0.5579     0.9613         68        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.63it/s]
                   all        712       3335      0.906      0.795       0.92      0.783

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/50      2.45G     0.7444     0.5475     0.9584         46        640: 100%|██████████| 366/366 [01:20<00:00,  4.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.67it/s]
                   all        712       3335      0.895      0.806      0.922      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/50      2.35G     0.7511     0.5463     0.9588         36        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.63it/s]
                   all        712       3335      0.778      0.891      0.929      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/50      2.38G     0.7472     0.5376     0.9568         76        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.23it/s]
                   all        712       3335      0.901      0.831      0.948      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      31/50       2.4G     0.7253     0.5258     0.9512         88        640: 100%|██████████| 366/366 [01:20<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.64it/s]
                   all        712       3335      0.813      0.912      0.958      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      32/50      2.49G     0.7208     0.5153      0.946         53        640: 100%|██████████| 366/366 [01:20<00:00,  4.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.64it/s]
                   all        712       3335      0.865      0.904      0.949      0.821

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      33/50      2.45G     0.7208     0.5169     0.9489         68        640: 100%|██████████| 366/366 [01:20<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.58it/s]
                   all        712       3335      0.823      0.881      0.932      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      34/50      2.45G     0.7117      0.508     0.9477         54        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.58it/s]
                   all        712       3335      0.796      0.934      0.948      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      35/50      2.46G     0.7195      0.505     0.9441         58        640: 100%|██████████| 366/366 [01:19<00:00,  4.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.62it/s]
                   all        712       3335      0.869      0.919      0.972      0.845

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      36/50      2.39G     0.7124     0.5004     0.9435         78        640: 100%|██████████| 366/366 [01:20<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.65it/s]
                   all        712       3335      0.902      0.902       0.97      0.845

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      37/50      2.43G     0.7077     0.4914     0.9417         67        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.24it/s]
                   all        712       3335      0.826      0.964       0.97      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      38/50      2.37G     0.6958     0.4802     0.9373         66        640: 100%|██████████| 366/366 [01:20<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.58it/s]
                   all        712       3335      0.821      0.964      0.969      0.849

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      39/50      2.42G     0.7009     0.4825     0.9366         74        640: 100%|██████████| 366/366 [01:19<00:00,  4.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.44it/s]
                   all        712       3335      0.858      0.961      0.975      0.853

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      40/50      2.43G     0.6997     0.4754     0.9401         49        640: 100%|██████████| 366/366 [01:20<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.65it/s]
                   all        712       3335      0.861      0.937      0.966      0.846
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      41/50      2.26G     0.6717      0.409     0.9164         25        640: 100%|██████████| 366/366 [01:19<00:00,  4.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.35it/s]
                   all        712       3335      0.882       0.95      0.973      0.853

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      42/50      2.25G     0.6615     0.3956     0.9106         28        640: 100%|██████████| 366/366 [01:19<00:00,  4.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.72it/s]
                   all        712       3335      0.898      0.948      0.976      0.855

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      43/50      2.25G     0.6555     0.3867     0.9102         51        640: 100%|██████████| 366/366 [01:18<00:00,  4.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.65it/s]
                   all        712       3335      0.895      0.929      0.966       0.85

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      44/50      2.25G     0.6471     0.3792     0.9055         58        640: 100%|██████████| 366/366 [01:19<00:00,  4.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.62it/s]
                   all        712       3335      0.927      0.922      0.975      0.859

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      45/50      2.24G     0.6455     0.3766     0.9044         36        640: 100%|██████████| 366/366 [01:19<00:00,  4.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.53it/s]
                   all        712       3335       0.94      0.931      0.981      0.869

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      46/50      2.24G     0.6359     0.3693     0.9005         28        640: 100%|██████████| 366/366 [01:19<00:00,  4.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.58it/s]
                   all        712       3335      0.917      0.948       0.98      0.863

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      47/50      2.24G     0.6333     0.3647     0.8999         29        640: 100%|██████████| 366/366 [01:19<00:00,  4.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.71it/s]
                   all        712       3335      0.945       0.92      0.982      0.867

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      48/50      2.25G     0.6296     0.3604     0.9002         29        640: 100%|██████████| 366/366 [01:19<00:00,  4.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.63it/s]
                   all        712       3335      0.935      0.958      0.985       0.87

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      49/50      2.27G     0.6236     0.3569     0.8955         39        640: 100%|██████████| 366/366 [01:19<00:00,  4.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:04<00:00,  4.60it/s]
                   all        712       3335      0.926       0.96      0.984       0.87

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      50/50      2.25G     0.6217     0.3547     0.8956         44        640: 100%|██████████| 366/366 [01:19<00:00,  4.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:05<00:00,  4.44it/s]
                   all        712       3335      0.929      0.959      0.985      0.869

50 epochs completed in 1.203 hours.
Optimizer stripped from runs\detect\train10\weights\last.pt, 5.5MB
Optimizer stripped from runs\detect\train10\weights\best.pt, 5.5MB

Validating runs\detect\train10\weights\best.pt...
Ultralytics 8.3.77 🚀 Python-3.12.3 torch-2.6.0+cu126 CUDA:0 (NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144MiB)
YOLO11n summary (fused): 100 layers, 2,584,102 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 23/23 [00:06<00:00,  3.74it/s]
                   all        712       3335      0.935      0.958      0.985       0.87
                  bike        177        448      0.954      0.929      0.977      0.825
    bike_bike_accident         34         34      0.906      0.912      0.974      0.872
  bike_person_accident         15         15      0.901          1      0.995      0.881
                   car        546       1996       0.97      0.965       0.99       0.85
     car_bike_accident         31         31      0.835          1      0.991       0.89
      car_car_accident        313        322      0.978      0.968      0.993      0.922
   car_object_accident         33         33      0.916       0.97      0.974      0.913
   car_person_accident          5          5          1      0.977      0.995      0.959
                person        185        451      0.951      0.899      0.975       0.72
Speed: 0.2ms preprocess, 2.8ms inference, 0.0ms loss, 1.0ms postprocess per image
Results saved to runs\detect\train10
💡 Learn more at https://docs.ultralytics.com/modes/train
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode

(Accident_Detection_and_Alert_System_Using_Yolov11+GAN) D:\Accident_Detection_and_Alert_System_Using_Yolov11+GAN>