# Toad re-identification


## git lfs

Model weights are storeds at `api/` directory.

`git lfs install`
`git lfs pull`

to load large files

## Training scripts

Training scripts and weights are located at `train` directory.

## How to run demo-webapp:

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

In the project directory, you should run:

### React-frontend

`yarn start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### Flask restAPI

In a separate terminal
`cd api`
`./start.sh`

Depends on pytorch, yolov5, torchvision, flask, PIL, cv2

Sample images for testing avaliable at test_crop
