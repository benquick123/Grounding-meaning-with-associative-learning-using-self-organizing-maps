#%%
# import things
from plotting import *
import pandas as pd
get_ipython().magic('matplotlib inline')
plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["figure.figsize"] = (9, 5)

train_type = ["position", "size", "color", "shape"]
_train_type = ["position", "size", "color", "type"]
log_path = "log-files/Jan-23_17.59.34_700_boot/log.csv"
save_path = "Plotting/" + log_path.split("/")[1] + "/"
dataframe = pd.read_csv(log_path)
mode = "show"


#%%
# plot different errors
errors = ["SOM error", "HEBB distances"]
underscores = [False, True]
ylims = (None, None)
for i, (e, u) in enumerate(zip(errors, underscores)):
    labels = [tr + " " + e for tr in train_type]
    labels1 = labels
    if u:
        labels = ["_" + l for l in labels]
    to_plot = [dataframe[l] for l in labels]
    labels = [tr + " " + e for tr in _train_type]
    plot_per_epoch(to_plot, labels, ylim=ylims[i], mode=mode, path=save_path + errors[i])


#%%
# plot wm and mw entropies
l_mw = []
l_wm_1 = []
ylims = ((0.7, 1.001), (0.8, 1.001), (0.6, 1.001), (0.6, 1.001))
for i, tr in enumerate(train_type):
    l_wm = []
    for tr1 in train_type:
        l_wm.append(("_" if tr != tr1 else "") + tr + " wm entropies - " + tr1)
        # l_mw.append(("_" if tr != tr1 else "") + tr + " mw entropies - " + tr1)
    l_mw.append(tr + " mw entropies")
    l_wm_1.append(tr + " wm entropies - " + tr)
    
    # to_plot_wm = [dataframe[l] for l in l_wm]
    # l_wm = [l[1:] if l[0] == "_" else l for l in l_wm]
    # plot_per_epoch(to_plot_wm, l_wm, ylim=ylims[i])
    
    # to_plot_mw = [dataframe[l] for l in l_mw]
    # l_mw = [l[1:] if l[0] == "_" else l for l in l_mw]
    # plot_per_epoch(to_plot_mw, l_mw)

to_plot_mw = [dataframe[l] for l in l_mw]
l_mw = [l + " V --> L entropies" for l in _train_type]
plot_per_epoch(to_plot_mw, l_mw, ylim=(0.7, 1.001), mode=mode, path=save_path + "mw_entropies")

to_plot_wm = [dataframe[l] for l in l_wm_1]
l_wm_1 = [l + " L --> V entropies" for l in _train_type]
plot_per_epoch(to_plot_wm, l_wm_1, ylim=(0.7, 1.001), mode=mode, path=save_path + "wm_entropies")


#%%
# probability tables entropies
ylim = (.0, 1.01)
l1 = [tr + " word_x_category entropy" for tr in train_type]
l2 = ["length: " + str(i+1) + " length_x_position entropy" for i in range(4)]
to_plot = [dataframe[l] for l in l1]
l1 = [tr + " word_x_category entropy" for tr in _train_type]
plot_per_epoch(to_plot, l1, ylim=ylim, mode=mode, path=save_path + "wxc")
to_plot = [dataframe[l] for l in l2]
plot_per_epoch(to_plot, l2, ylim=ylim, mode=mode, path=save_path + "lxp")


#%%



