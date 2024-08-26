import glob
import os
import time

import nest
import numpy as np
import pandas as pd


class Tools:
    def __init__(self, parser):
        self.timing = []
        self.parser = parser
        self.args = parser.parse_args()

    def time(self):
        self.timing.append(time.time())

    def select(self, weights, sign):
        condition = weights >= 0.0 if sign >= 0.0 else weights < 0.0
        post, pre = np.where(condition)
        return post, pre

    def apply_dales_law(self, n_pop, n_nrns, label, weights):
        if label in self.args.apply_dales_law:
            n_per_pop = n_nrns // n_pop
            for i in range(n_pop):
                start = i * n_per_pop
                half = start + n_per_pop // 2
                end = start + n_per_pop
                weights[:, start:half] = np.abs(weights[:, start:half])
                weights[:, half:end] = -np.abs(weights[:, half:end])
        return weights

    def constrain_weights(
        self,
        nrns_in,
        nrns_rec,
        nrns_out,
        weights_in_rec,
        weights_rec_rec,
        weights_rec_out,
        params_syn_base,
        params_common_syn_eprop,
    ):
        weight_dicts = [
            {"label": "in_rec", "weights": weights_in_rec, "nrns_pre": nrns_in, "nrns_post": nrns_rec},
            {"label": "rec_rec", "weights": weights_rec_rec, "nrns_pre": nrns_rec, "nrns_post": nrns_rec},
            {"label": "rec_out", "weights": weights_rec_out, "nrns_pre": nrns_rec, "nrns_post": nrns_out},
        ]

        sign_dicts = [
            {"Wmin": 0.0, "Wmax": 100.0},
            {"Wmin": -100.0, "Wmax": 0.0},
        ]

        weights_list, nrns_pre_list, nrns_post_list = [], [], []
        for weight_dict in weight_dicts:
            n_pre = len(weight_dict["nrns_pre"].tolist())
            weight_dict["weights"] = self.apply_dales_law(1, n_pre, "in_rec", weight_dict["weights"])
            if weight_dict["label"] in self.args.prevent_weight_sign_change + self.args.apply_dales_law:
                conns = nest.GetConnections(weight_dict["nrns_pre"], weight_dict["nrns_post"])
                nest.Disconnect(conns)
                weights_list.append(weight_dict["weights"])
                nrns_pre_list.append(np.array(weight_dict["nrns_pre"]))
                nrns_post_list.append(np.array(weight_dict["nrns_post"]))

        if len(self.args.prevent_weight_sign_change + self.args.apply_dales_law) > 0:
            for sign_dict in sign_dicts:
                sign = np.sign(sign_dict["Wmin"])
                pop_post_arr = np.array([], dtype=int)
                pop_pre_arr = np.array([], dtype=int)
                weights_arr = np.array([])
                for weights, nrns_pre, nrns_post in zip(weights_list, nrns_pre_list, nrns_post_list):
                    post, pre = self.select(weights, sign)
                    pop_pre_arr = np.append(pop_pre_arr, np.array(nrns_pre)[pre])
                    pop_post_arr = np.append(pop_post_arr, np.array(nrns_post)[post])
                    weights_arr = np.append(weights_arr, weights[post, pre].flatten())

                base_synapse_model = params_syn_base["synapse_model"]

                params_common = params_common_syn_eprop
                params_common["optimizer"].update(sign_dict)
                params_common["weight"] = sign_dict["Wmin"]
                synapse_model = f"{base_synapse_model}_{sign}"
                nest.CopyModel(base_synapse_model, synapse_model, params_common)

                params_base = params_syn_base.copy()
                params_base["synapse_model"] = synapse_model
                params_base["weight"] = weights_arr
                params_base["delay"] = np.ones_like(weights_arr) * params_syn_base["delay"]

                nest.Connect(pop_pre_arr, pop_post_arr, conn_spec="one_to_one", syn_spec=params_base)

    def save_weights_snapshots(self, weights_dict_pre, weights_dict_post):
        for phase, weights_dict in zip(["pre", "post"], [weights_dict_pre, weights_dict_post]):
            for k, v in weights_dict.items():
                label = f"{phase}_train_{k}"
                v_ = {k: v[k] for k in v.keys() if k != "weight_matrix"}
                pd.DataFrame.from_dict(v_).to_csv(f"{self.args.recordings_dir}/weights_{label}.csv", index=False)

    def save_weight_recordings(self, sender_nrns, target_nrns, df, label, recorder_label):
        sender_min = min(sender_nrns.tolist())
        sender_max = max(sender_nrns.tolist())
        target_min = min(target_nrns.tolist())
        target_max = max(target_nrns.tolist())

        condition1 = (df.sender >= sender_min) & (df.sender <= sender_max)
        condition2 = (df.targets >= target_min) & (df.targets <= target_max)

        df_sub = df[condition1 & condition2]

        if label != "":
            label = f"_{label}"
        df_sub.to_csv(f"{self.args.recordings_dir}/{recorder_label}{label}.csv", index=False)

    def process_recordings(self, duration, nrns_in, nrns_rec, nrns_out):
        recorder_labels = ["multimeter_out", "spike_recorder_in", "weight_recorder"]
        if "evidence" in self.parser.prog:
            nrn_types = ["reg", "ad"]
        else:
            nrn_types = ["rec"]
        for nrn_type in nrn_types:
            recorder_labels.extend([f"multimeter_{nrn_type}", f"spike_recorder_{nrn_type}"])

        for recorder_label in recorder_labels:
            save_file = f"{self.args.recordings_dir}/{recorder_label}"

            if os.path.exists(f"{save_file}.csv"):
                df = pd.read_csv(f"{save_file}.csv")
            else:
                df = pd.DataFrame()

            for fname in sorted(glob.glob(f"{save_file}*.dat")):
                df = pd.concat([df, pd.read_csv(f"{fname}", skiprows=2, sep="\t")], ignore_index=True)
                os.remove(fname)

            if recorder_label == "weight_recorder":
                self.save_weight_recordings(nrns_in + nrns_rec, nrns_rec + nrns_out, df, "", recorder_label)
                self.save_weight_recordings(nrns_in, nrns_rec, df, "in", recorder_label)
                self.save_weight_recordings(nrns_rec, nrns_rec, df, "rec", recorder_label)
                self.save_weight_recordings(nrns_rec, nrns_out, df, "out", recorder_label)

            else:
                df.to_csv(f"{save_file}.csv", index=False)

                condition1 = (df["time_ms"] > 0) & (df["time_ms"] < duration["sequence"] + 50)
                condition2 = (df["time_ms"] > duration["task"] - duration["sequence"] - 50) & (
                    df["time_ms"] < duration["task"]
                )

                df_subset = df[condition1 | condition2]
                df_subset.to_csv(f"{save_file}_subset.csv", index=False)

    def process_timing(self, kernel_status):
        timing_dict = {}
        for k, v in kernel_status.items():
            if k.startswith("time") or k == "biological_time":
                timing_dict[k] = v
        pd.DataFrame([timing_dict]).to_csv(f"{self.args.recordings_dir}/timing.csv", index=False)

    def get_events(self, label):
        df = pd.read_csv(f"{self.args.recordings_dir}/{label}.csv")

        events = {}
        for k in df.columns:
            values = np.array(df[k])
            if k == "sender":
                k = "senders"
            elif k == "time_ms":
                k = "times"
            events[k] = values
        return events

    def save_performance(self, performance_dict):
        self.loss = performance_dict["loss"]
        pd.DataFrame.from_dict(performance_dict).to_csv(
            f"{self.args.recordings_dir}/learning_performance.csv", index=False
        )

    def verify(self):
        file_name = self.parser.prog
        if file_name == "eprop_supervised_classification_evidence-accumulation_bsshslm_2020.py":
            loss_reference = [
                0.74115255000619,
                0.74038818770074,
                0.66578523317777,
                0.66364419332299,
                0.72942896284495,
            ]
        elif file_name == "eprop_supervised_classification_evidence-accumulation.py":
            loss_reference = [
                33.99707684974502,
                34.54963564066860,
                36.15785645581101,
                35.49256076087588,
                35.86926256681858,
            ]
        elif file_name == "eprop_supervised_regression_sine-waves_bsshslm_2020.py":
            loss_reference = [
                101.96435699904158,
                103.46673112620579,
                103.34060707477168,
                103.68024403768638,
                104.41277574875247,
            ]
        elif file_name == "eprop_supervised_regression_sine-waves.py":
            loss_reference = [
                102.32452148947576,
                102.02905791284471,
                104.18266745765692,
                104.67929923397385,
                105.13095153301973,
            ]
        elif file_name == "eprop_supervised_classification_neuromorphic_mnist.py":
            loss_reference = [
                0.49165621803559,
                0.50931366438794,
                0.50123237065641,
                0.49169665763603,
                0.45463618073378,
            ]
        elif file_name == "eprop_supervised_regression_lemniscate_bsshslm_2020.py":
            loss_reference = [
                314.30442538643001,
                313.84127193622919,
                312.33971633807948,
                310.66410755892281,
                309.19353500432857,
            ]
        elif file_name == "eprop_supervised_regression_handwriting_bsshslm_2020.py":
            loss_reference = [
                91.40191610510351,
                90.53583357361666,
                89.91415022333089,
                88.54544175584950,
                86.98770239575573,
            ]

        n_compare = min(len(self.loss), len(loss_reference))
        verification_successful = np.allclose(self.loss[:n_compare], loss_reference[:n_compare], atol=1e-14)

        if not verification_successful:
            for l, lr in zip(self.loss[:n_compare], loss_reference[:n_compare]):
                print(f"    {l:.14f},")
                print(f"    {lr:.14f},")
                print(" ")
        print(verification_successful)
