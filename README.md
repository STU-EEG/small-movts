# small-movts
small movements detection from video

## reproducing training

1. download and unpack the archive: https://figshare.com/articles/dataset/blink_non-blink_events_on_video/32235561
1. clone this repository
1. train CNN and other archs:
    ```shell
    micromamba create -f environment.yml
    micromamba activate samll-movts
    python train.py
    ```
1. optionally: evaluate CNN
    ```shell
    python eval.py
    ```


## reproducing the whole pipeline (given original, private-hosted data)

1. pose detection with YOLOv5
    ```shell
    micromamba create -f ultralytics.yml
    micromamba activate ultralytics-env
    python yolo-server.py
    ```
1. crop blinks from video
    ```shell
    # TODO ensure consistency
    micromamba activate video_eeg # TODO where it originates?
    python crop_video_blinks.py
    ```
1. sort manually
    ```shell
    micromamba activate small-movts
    python src/sort_videos.py -i untracked/vkuk -o untracked/vkuk
    ```
1. train/test split:
    ```shell
    ./train_test_split.sh
    ```
1. CNN training (see section above)

