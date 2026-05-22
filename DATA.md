# mimic-video training data

We are using two independent dataloading pipelines for loading either video only or video plus associated action data.

| Training Stage          | Video Data Required | Action Data Required | Data Source |
| ----------------------- | ------------------- | -------------------- | ----------- |
| Video Model Finetuning  | Yes                 | No                   | mp4         |
| Action Decoder Training | Yes                 | Yes                  | zarr        |

## Video

We use a dataloading implementation based on the existing video finetuning dataloading of Cosmos-Predict2. Videos are stored as one `mp4` file per episode in `dataset_name/video/ep.mp4` while language instructions are stored in an associated `txt` file in `dataset_name/metas/ep.txt`. Language embeddings are precomputed and stored in `dataset_name/t5_xxl/ep.pickle`.

We sample uniformly from the set of episodes and choose a time `t` uniformly from `[0, N + H_O]` where `N` is the number of frames in the episode and `H_O` is the observation horizon. We then load the video frames from `t-H_O` to `t+H_P` where `H_P` is the prediction horizon. We pad the episode on either end by repeating the first resp. last frame.

Thus, the earliest frame sequence that may be sampled from an episode is `[0, 0, 0, 0, 0, 1, 2, 3, ..., min(N, 56)]` and the latest frame sequence that may be sampled is `[-1, -1, -1, ..., -1]`.

Given a target frequency `f`, for the `i`th sequence element we choose the video frame closest to `t + i / f`.

The implementation can be found in [dataset_video.py](../model/cosmos_predict2/data/dataset_video.py).

An overview over the scripts in `video` is given here
```
data_preprocessing
├── action
└── video
    ├── extract_val_snippets.py           # extract example snippets from the validation set to new mp4s.
    ├── get_t5_embeddings.py              # create t5_xxl/ with language embeddings for any dataset.
    ├── process_bridge_video_and_lang.py  # create video/ and metas/ from the raw bridge dataset.
    ├── process_libero_lang.py            # create metas/ for the libero datasets, assuming video/ already exists.
    └── process_libero_video.py           # create video/ for the libero datasets from h5.
```

## Action

Our full video + action dataloading pipeline stores both video and action data in two `zarr` arrays per data key (primary image, end effector pose, end effector joint angle(s), ...) per episode. One array contains the raw sequence of values while the other contains the corresponding timestamps.

To assemble a training sample, we first uniformly choose a time `t` from the timestamps of the primary video. For each modality, we define a horizon and a target frequency and use the observed timestamp data to interpolate the raw values to the desired times. We pad episodes by repeating the first resp. last values to enable horizons extending further than the episode boundaries. The implementation of reading from the zarr files and interpolating the values according to the specification can be found in [chunk_reader.py](./model/cosmos_predict2/data/action/chunk_reader.py).

After interpolation, we transform all predicted poses to be relative to the current proprioceptive pose and express all rotations using the first two rows of their matrix representation. The collection of data transforms is in [data_transforms.py](./model/cosmos_predict2/data/action/data_transforms.py). The torch dataset bringing it all together by calling the `chunk_reader` and subsequently applying transformations is defined in [dataset_action.py](./model/cosmos_predict2/data/action/dataset_action.py).

We further normalize all non-rotation non-video values component-wise by clipping outliers, removing the mean, and rescaling to unit dataset variance (in that order). The normalization code lives in [normalizer.py](./model/cosmos_predict2/module/normalizer.py).

For information on how the video-action dataloading is configured, see [MODEL.md](./MODEL.md).

The batch will then be a flat dict with the following keys:
- `obs/language_embedding`
- `obs/lowdim_concat`: proprioception, flattened and concatted
- `obs/workspace_rgb`: 5 frames of history
- `action/lowdim_concat`: actions, flattened and concatted
- `action/workspace_rgb`: 56 frames of future

An overview over the preprocessing scripts in `data_preprocessing/action` is given here
```
data_preprocessing
├── action
│   ├── precompute_t5.py      # add language_embedding to zarrs assuming language_instruction exists.
│   ├── process_bridge.py     # create zarr dataset from raw bridge dataset.
│   ├── process_libero.py     # create zarr dataset from h5 libero dataset.
│   ├── regenerate_libero.py  # replay and filter + process libero dataset without changing format.
│   └── subsample_zarr.py     # sample random per-task subset of libero zarr dataset.
└── video
```
