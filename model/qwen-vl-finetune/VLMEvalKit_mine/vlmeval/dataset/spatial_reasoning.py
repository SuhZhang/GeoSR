from huggingface_hub import snapshot_download
from pathlib import Path
from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

FAIL_MSG = 'Failed to obtain answer via API.'

GEOSR4D_ROOT = Path(__file__).resolve().parents[5]
GEOSR4D_DATA_ROOT = Path(
    os.environ.get("GEOSR4D_DATA_ROOT", GEOSR4D_ROOT / "data")
)


def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


class SPATIAL_REASONING(VideoBaseDataset):

    SYS = ''

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='Spatial-Reasoning', use_subtitle=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_subtitle = use_subtitle
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['Spatial-Reasoning']

    def prepare_dataset(self, dataset_name='Spatial-Reasoning'):
        dataset_path = str(
            Path(
                os.environ.get(
                    "GEOSR4D_BENCH_VIDEO_ROOT",
                    GEOSR4D_DATA_ROOT / "spatial_reasoning" / "videos_bench",
                )
            )
        )
        parquet_path = str(
            Path(
                os.environ.get(
                    "GEOSR4D_BENCH_PARQUET",
                    GEOSR4D_DATA_ROOT / "spatial_reasoning" / "benchmark.parquet",
                )
            )
        )

        def generate_tsv(pth):
            tsv_path = osp.join("./", f"{dataset_name}.tsv")
            audit_dir = osp.join("./", f"{dataset_name}_audit")
            os.makedirs(audit_dir, exist_ok=True)

            # 如果你希望每次都重新生成并过滤（推荐调试期这样做），把下面两行打开：
            # if os.path.exists(tsv_path):
            #     os.remove(tsv_path)

            # 如果你仍想“存在就不重新生成”，保留原逻辑：
            if os.path.exists(tsv_path):
                return

            if not osp.exists(parquet_path):
                raise FileNotFoundError(
                    f"Benchmark parquet not found: {parquet_path}. "
                    "Set GEOSR4D_BENCH_PARQUET to the correct file."
                )

            df = pd.read_parquet(parquet_path)
            df = df.assign(index=range(len(df)))
            df["video"] = df["videoID"]
            df["video_path"] = df["videoID"].apply(lambda x: f"{dataset_path}/{x}.mp4")
            df["candidates"] = df["options"].apply(lambda x: x.tolist())

            # ---------- 新增：过滤缺失视频 ----------
            # 只保留 mp4 存在的样本
            df["video_exists"] = df["video_path"].apply(os.path.exists)

            missing_df = df.loc[~df["video_exists"], ["index", "videoID", "video", "video_path"]].copy()
            kept_df = df.loc[df["video_exists"]].copy()

            audit = {
                "N_total": int(len(df)),
                "N_eval": int(len(kept_df)),
                "N_missing": int(len(missing_df)),
                "coverage": float(len(kept_df) / len(df)) if len(df) else 0.0,
                "dataset_path": dataset_path,
                "parquet_path": parquet_path,
                "missing_examples": missing_df.head(50).to_dict(orient="records"),
            }

            # 保存 missing 全列表 & summary
            with open(osp.join(audit_dir, "missing_videos.json"), "w") as f:
                json.dump(missing_df.to_dict(orient="records"), f, indent=2)
            with open(osp.join(audit_dir, "coverage_summary.json"), "w") as f:
                json.dump(audit, f, indent=2)

            # 清理辅助列
            kept_df = kept_df.drop(columns=["video_exists"])
            # ---------- 新增结束 ----------

            kept_df = kept_df[["index", "video", "video_path", "candidates", "question", "answer"]]
            kept_df.to_csv(tsv_path, sep="\t", index=False)

            print(
                f"[Spatial-Reasoning TSV] total={audit['N_total']} eval={audit['N_eval']} "
                f"missing={audit['N_missing']} coverage={audit['coverage']:.3f} "
                f"audit_dir={audit_dir}"
            )
        generate_tsv(dataset_path)
        data_file = osp.join('./', f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):

        vid_path = osp.join(self.data_root, video + '.mp4')
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm)

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, line['video'] + '.mp4')))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        text_prompt = self.FRAMES_TMPL_NOSUB
        message.append(dict(type='text', value=text_prompt))
        line['question'] += '\n' + '\n'.join(eval(line['candidates']))
        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.videomme import get_dimension_rating, extract_characters_regex, extract_option

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                if extract_characters_regex(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'Video-MME'
                    )
                    data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[data['index'] == idx, 'score'] = int(extract_characters_regex(pred) == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        #rating = get_dimension_rating(score_file)
        #dump(rating, tgt_file)
        return None
