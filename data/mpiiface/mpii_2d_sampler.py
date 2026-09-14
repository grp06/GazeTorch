import numpy as np

EPS = 1e-8
EPS_INV = 1e8
class MPII_2D_Sampler:
    def __init__(self, id_list:list, gaze_list, target_num,
                 nbins_yaw=10, bin_size_yaw=4, yaw_bias=-20,
                 nbins_pitch=8, bin_size_pitch=3, pitch_bias=-22.5, **kwargs):
        self.id_list_original = id_list.copy()
        self.gaze_list = gaze_list

        self.n = len(id_list)
        self.n_target = target_num

        self.nbins_yaw = nbins_yaw
        self.bin_size_yaw = bin_size_yaw
        self.yaw_bias = yaw_bias
        self.nbins_pitch = nbins_pitch
        self.bin_size_pitch = bin_size_pitch
        self.pitch_bias = pitch_bias

        self.init_histogram()
        self.quote_bin_pop = self.get_quote()
        print("[MPII_Sampler]Init done.")

    def init_histogram(self):
        self.H = np.zeros((self.nbins_pitch, self.nbins_yaw))
        self.bin_id = np.empty_like(self.gaze_list)

        self.bins = []
        for i in range(self.nbins_pitch):
            self.bins.append([])
            for _ in range(self.nbins_yaw):
                self.bins[i].append([])

        for i in range(self.n):
            g = self.gaze_list[i] * 180 / np.pi
            pitch_bin_id = np.clip( int((g[0] - self.pitch_bias) / self.bin_size_pitch), 0, self.nbins_pitch-1)
            yaw_bin_id = np.clip(int((g[1] - self.yaw_bias) / self.bin_size_yaw), 0, self.nbins_yaw-1)
            self.H[pitch_bin_id, yaw_bin_id] += 1
            self.bin_id[i,0] = pitch_bin_id
            self.bin_id[i,1] = yaw_bin_id
            self.bins[pitch_bin_id][yaw_bin_id].append(self.id_list_original[i])
        return self.H

    def get_quote(self):
        self.H = self.H.astype(int)
        ans = np.ceil(np.ones_like(self.H) * (self.n_target / (self.nbins_pitch * self.nbins_yaw))).astype(int)
        return ans

    def sample(self):
        new_id_list = []
        for pitch in range(self.nbins_pitch):
            for yaw in range(self.nbins_yaw):
                if self.quote_bin_pop[pitch, yaw]==0 or self.H[pitch, yaw]==0:
                    continue
                n_repeat = self.quote_bin_pop[pitch, yaw] // self.H[pitch, yaw]
                n_tail = self.quote_bin_pop[pitch, yaw] % self.H[pitch, yaw]
                sampled_bin = []
                for i in range(n_repeat):
                    sampled_bin = np.append(sampled_bin, self.bins[pitch][yaw])
                sampled_bin = np.append(sampled_bin,
                                        np.random.permutation(self.bins[pitch][yaw])[:n_tail])
                new_id_list = np.append(new_id_list, sampled_bin)
        return np.random.permutation(new_id_list).astype(int)