# AVIRI-demo
This is the simple AI demo code for Disease-related Visual Impairment Using Retinal Imaging

## The process of our software is as below:
1) start mmd-vi-app.service service,  or manually run our python software directly 
2) SW check if there is any image under “./images” folder
3) our pre-trained model works and  test the image, then delete the input image file
4) The prediction output will be saved in the “./outputs/predictions.csv”, the file keep only one record for current input image, the information will be used to display in web APP.

## VI project
The live demo can refer to https://modstore.org/VI
The current backend code is in 172.20.116.162 server, front-end APP is in https://modstore.org/VI/
images and outputs folder will be created automatically when you run the python file
 
```bash
Login with:
ssh user@172.20.116.162
 
user@m10:~/AVIRI-demo$ tree -L 1
.
├── best_model.h5
├── imagenet_utils.py
├── images
├── main_vi.py
├── mmd-vi-app.service
├── outputs
├── README.md
├── samples
└── vi_run.sh
```
 
Some functionality supported in the web application: https://modstore.org/VI/
1) The web app title changes to   -- Analysis of Pathological Visual Impairment
2) the input image format support  jpg, bmp, tif, png and dcm -- Please upload fundus photo in jpeg, png, tiff, bmp or dcm format.
3) replace the samples images – there are 20 images in samples folder – Note: please show the original name in sample folder to web UI.
4) The output is in ./outputs folder, the format is as attach .csv file, the probability can be shown as
Absence of Pathological Visual Impairment (Probability):
Presence of Pathological Visual Impairment (Probability): 
5) the biggest difference is that --  As one part of the result, web app need to display the output heatmap image under “./outputs/heatmap.jpg”, the example of the output image is as attached.
 
### Running the example manually
```bash
user@m10:~/AVIRI-demo$ python3 main_vi.py 
Loaded checkpoint 'best_model.h5'.
loading 1 model, time 2.38
Input Image:  IS34507_AV00(SOZ.001.JPG
Softmax output:  [[0.9818627  0.01813722]]
[VI_CNN: Threshold(0.0131) / Prediction: 0.9819]  probability: Normal: 0.3611; VI: 0.6389
Input Image:  IS39448_2_0.jpg
Softmax output:  [[0.9988424  0.00115765]]
[VI_CNN: Threshold(0.0131) / Prediction: 0.9988]  probability: Normal: 0.9558; VI: 0.0442
```

## Start and stop the service for demo application
Normally the service file is under the system directory as below
```bash
/etc/systemd/system/*
/run/systemd/system/*
/lib/systemd/system/*
```

you can find the service files in backend server
```bash
ls /etc/systemd/system/
mmd-vi-app.service
```

The content of service file is as below(example only, the folder and running app name could be different based on the setup)
```bash
cat /etc/systemd/system/mmd-vi-app.service
[Unit]
Description=MMD VI App
After=syslog.target network.target

[Service]
WorkingDirectory=/home/user/project/xuxinx/VI
ExecStart=/home/user/project/xuxinx/VI/vi_run.sh
User=user
KillMode=process
Restart=on-failure
RestartSec=30s

[Install]
WantedBy=multi-user.target

[Install]
WantedBy=multi-user.target
```

You can start or stop service as below
```bash
sudo systemctl start mmd-vi-app.service
sudo systemctl stop mmd-vi-app.service
```
