from sklearn.metrics import confusion_matrix
import bqplot.pyplot as plt
import matplotlib.pyplot as mplt
import bqplot as bq
import ipywidgets as widgets
import numpy as np
import bqplot.pyplot
from sklearn.linear_model import LogisticRegression


DPI = 110  # for plot fonts etc


def show_cohort(y_true, protected, backend="bqplot"):
    """Show the base rates in the cohort by protected attribute."""

    # jerry-rig confusion matrix to count prevalence in the population
    mn, fn, mp, fp = confusion_matrix(y_true, protected).ravel()
    # Now we can do two histograms:
    values = [[mp, fp], [mn, fn]]
    colors = ["green", "red"]

    if backend == "bqplot":
        fig = plt.figure(min_aspect_ratio=1, max_aspect_ratio=1)
        # First index is colour, second index is X

        # Note - putting negative does cool weird stuff
        bars = plt.bar(
            x_ticks, values,
            colors=colors,
            # display_legend=False,
            # labels=["Good Customers", "Bad Customers"],
        )
        siz = "4in"
        fig.layout.width = siz
        fig.layout.height = siz

    else:
        fig, ax = mplt.subplots(figsize=(5, 5), dpi=DPI)
        ax.bar(x_ticks, values[0], color=colors[0])
        ax.bar(x_ticks, values[1], bottom=values[0], color=colors[1])
    return fig


def metrics(id, y_true, y_pred, protected, desc, x_ticks, pname, backend="bqplot"):
    """Plot a stack of figures for the three metrics."""
    figure_name = f"Fairness_{id}_{desc}"

    dis = protected.astype(bool)
    tn_adv, fp_adv, fn_adv, tp_adv = confusion_matrix(
        y_true[~dis], y_pred[~dis]
    ).ravel()

    tn_dis, fp_dis, fn_dis, tp_dis = confusion_matrix(
        y_true[dis], y_pred[dis]
    ).ravel()

    # Also just count all positives and negatives
    pos_adv = tp_adv + fp_adv  # all positives
    pos_dis = tp_dis + fp_dis
    neg_adv = tn_adv + fn_adv  # all negatives
    neg_dis = tn_dis + fn_dis

    # Demographic Parity
    dem_par = (pos_dis / (pos_dis + neg_dis)) / (pos_adv / (pos_adv + neg_adv))

    # Equal Opportunity
    eq_opp = (tp_dis / (tp_dis + fn_dis)) / (tp_adv / (tp_adv + fn_adv))

    acc_adv = np.mean(y_true[~dis] == y_pred[~dis])
    acc_dis = np.mean(y_true[dis] == y_pred[dis])
    print("Acc adv: {}".format(acc_adv))
    print("Acc dis: {}".format(acc_dis))

    def normalise(nd):
        numer, denom = nd
        numer = np.asarray(numer, float)
        denom = np.asarray(denom, float)
        total = numer + denom
        total /= 100.  # make percentage
        numer = numer / total
        denom = denom / total
        return [numer, denom]

    def describe(rate):
        rate = np.round(rate, 2)  # make a percent

        if rate < 0.5:
            return f"{rate:.0%} as often"
        elif rate == 0.5:
            return "half as often"
        elif rate < 1.:
            return f"{1. - rate:.0%} less often"
        elif rate == 1.:
            return "equally often"
        elif rate < 2.:
            return f"{rate - 1.:.0%} more often"
        elif rate == 2.:
            return "twice as often"
        elif rate < 4:
            return f"{rate:.1f}x as often"
        elif rate < 10:
            return f"{rate:.0f}x as often"
        else:
            return "significantly more often"
        # if eo > 0.995:
        #     stake = f"Suitable {groups[0].capitalize()} and {groups[1]} are selected at equal rates."

    def bqbars(values, colors, title, stake):
        fig = plt.figure()
        eo_bars = plt.bar(x_ticks, values, colors=colors)
        plt.ylabel("Fraction (%)")
        plt.xlabel(stake)
        fig.title = title
        return fig

    def matplotbars(ax, values, colors, title, stake):
        b0 = ax.bar(x_ticks, values[0], width=0.5, color=colors[0])
        b1 = ax.bar(x_ticks, values[1], bottom=values[0], width=0.5, color=colors[1])
        ax.set_ylabel("Fraction (%)")
        ax.set_xlabel(stake)
        ax.set_title(title)
        return b0, b1

    eov = normalise([[tp_adv, tp_dis], [fn_adv, fn_dis]])
    eov_title = "Opportunity{}".format(desc)
    eov_stake = f"Suitable {pname} are selected {describe(eq_opp)}."
    eov_colors = ["green", "lightgreen"]

    dpv = normalise([[pos_adv, pos_dis], [neg_adv, neg_dis]])
    dpv_title = "Selection Rates{}".format(desc)
    dpv_stake = f"{pname} are selected {describe(dem_par)}."
    dpv_colors = ["black", "gray"]

    ppv = normalise([[tp_adv, tp_dis], [fp_adv, fp_dis]])
    # error rates (false Discovery Rate parity)
    prec_par = (tp_dis / pos_dis) / (tp_adv / pos_adv)
    ppv_title = "Precision{}".format(desc)
    #ppv_stake = f"{pname.capitalize()} are selected {describe(prec_par)}.".replace("often", "precisely")
    ppv_stake = f"Selected {pname} are profitable {describe(prec_par)}."
    ppv_colors = ["green", "red"]


    print("Selection: {}".format(dpv))
    print("Opportunity: {}".format(eov))
    print("Precision: {}".format(ppv))
    if backend == "bqplot":
        scales = {
            "x": bq.OrdinalScale(),
            "y": bq.LinearScale(),
        }

        eo_fig = bqbars(eov, eov_colors, eov_title, eov_stake)
        dp_fig = bqbars(dpv, dpv_colors, dpv_title, dpv_stake)
        pp_fig = bqbars(ppv, ppv_colors, ppv_title, ppv_stake)

        fairbox = widgets.HBox((
            dp_fig,
            eo_fig,
            pp_fig,
        ))
        fairbox.layout.width = "99%"
        display(fairbox)
        eo_fig.axes[0].color = eo_fig.axes[1].color = "Black"
        dp_fig.axes[0].color = dp_fig.axes[1].color = "Black"
        pp_fig.axes[0].color = pp_fig.axes[1].color = "Black"

    else:
        fig, ax = mplt.subplots(1, 3, figsize=(9, 3), dpi=DPI)

        def clean(strv):
            # We have a bit more room for better sentences
            val = strv.replace("SE Asian", "SE Asians").replace("x ", " times ")
            # split the line
            s = len(val) // 2
            a = val[s:].find(' ')
            b = val[:s][::-1].find(' ')
            split = s + a
            if (b >= 0 and b < a) or (a < 0):
                split = s - b - 1
            val = val[:split] + "\n" + val[split + 1:]
            return val

        sa, na = matplotbars(ax[0], dpv, dpv_colors, dpv_title, clean(dpv_stake))
        es, en = matplotbars(ax[1], eov, eov_colors, eov_title, clean(eov_stake))
        _, ii = matplotbars(ax[2], ppv, ppv_colors, ppv_title, clean(ppv_stake))
        mplt.legend(
            (es, en, sa, na, ii),
            ("Profitable (Selected)",
             "Profitable (Rejected)",
             "Selected (All)",
             "Rejected (All)",
             "Not Profitable (Selected)",
             ),
            bbox_to_anchor=(1.25, 0.75),
        )
        mplt.tight_layout()
        fig.savefig("images/" + figure_name + ".png", bbox_inches="tight")
        # display(fig)


def mplt_bars(ax, ticks, values, colors, ylabel=None, title=None):
    """Quick function for creating stacked matplotlib barplot"""
    bar0 = ax.bar(ticks, values[0], color=colors[0])
    bar1 = ax.bar(ticks, values[1], bottom=values[0], color=colors[1])
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return bar0, bar1


def plot_feature_importances(scenarios, backend):
    # Plot all the feature importances
    for id in scenarios:
        # Make scenario data
        scenario = scenarios[id]
        desc = scenario.description
        figure_name = f"Importance_{id}_{desc}"
        train_data = scenario.gendata("train")
        test_data = scenario.gendata("test")
        np.random.seed(42)
        X_train, y_train = scenario.get_modelling_data(train_data)
        X_test, y_test = scenario.get_modelling_data(test_data)

        dis = scenario.ticks[1]
        if dis != "SE Asian proxy":  # proper noun
            dis = dis.lower()

        if "pinterest" in train_data.columns:
            ren = {
                "pinterest": scenario.proxy_name,
                'disadv_flag': dis,
            }
            # warning: train data seems to be writable in-place
            X_train = X_train.rename(columns=ren)
            X_test = X_test.rename(columns=ren)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        columns = list(X_train.columns)
        importance = clf.coef_[0]

        if True:
            # standardize?
            importance = importance / X_train.std(axis=0)

        heights = np.abs(importance)
        cols = ['orange', 'blue']
        colors = (np.array(cols)[(importance >= 0).astype(int)]).tolist()
        title = "Model Feature Importance"

        print("Importance: {}".format(importance))

        if backend == "matplotlib":
            fig, ax = mplt.subplots(figsize=(5, 5), dpi=DPI)
            ax.bar(columns, importance, color=colors)
            ax.axhline(0, color="Black", lw=1.)

            # Add some dummies for a legend
            c = columns[0]
            mplt.bar([c], [0], color=cols[1], label="Increases Score")
            mplt.bar([c], [0], color=cols[0], label="Decreases Score")
            mplt.plot([-0.5, len(columns) - 0.5], [0, 0], 'k')

            scale = 10
            if int(id) == 5:
                # this scenario uses huge weights for some reason...
                scale = 40

            mplt.ylim(-scale, scale)
            mplt.legend(bbox_to_anchor=(1.5, 0.5))

            ax.set_title(title)
            fig.savefig("images/" + figure_name + ".png", bbox_inches="tight", dpi=300)

        elif backend == "bqplot":
            fig = plt.figure(title=title, min_aspect_ratio=1, max_aspect_ratio=1)
            for c, h, colr in zip(columns, importance, colors):
                plt.bar([c], [h], colors=[colr])  # each bar is its own bar
            plt.ylim(-10, 10)  # was -10, 10 except for scenario 5?
            fig.axes[0].color = fig.axes[1].color = "Black"
            fig.layout.width = fig.layout.height = "5in"
            display(fig)


def plot_profitability_distributions(scenarios, backend):
    for id in scenarios:
        # Make scenario data
        scenario = scenarios[id]
        desc = scenario.description

        train_data = scenario.gendata("train")
        test_data = scenario.gendata("test")
        np.random.seed(42)
        X_train, y_train = scenario.get_modelling_data(train_data)
        X_test, y_test = scenario.get_modelling_data(test_data)
        data = {
            'training': (X_train, y_train, train_data.disadv_flag),
            'deployment': (X_test, y_test, test_data.disadv_flag),
        }

        figure_name = f"Profitability_{id}_{desc}"
        print(f"for Scenario {id}: {desc}")

        if backend == "matplotlib":
            l = len(data.items())
            fig, ax = mplt.subplots(1, l, figsize=(5 * l, 5), dpi=DPI)
            i = 0
        else:
            plts = []

        for cohort, (X, y_true, prot) in data.items():
            dis = prot.astype(int)
            mn, fn, mp, fp = confusion_matrix(y_true, dis).ravel()
            values = [[mp, fp], [mn, fn]]
            # half way between selected and not
            colors = ["#ff4045", "#48B748"]
            title = f"Representation in {cohort} cohort."
            ylabel = "Number of Applicants"

            if backend == "matplotlib":
                bars = mplt_bars(ax[i], scenario.ticks, values, colors, ylabel, title)
                mplt.legend(bars, ["profitable customers", "non-profitable customers"],
                            bbox_to_anchor=(1.05, 0.5)  # sit outside plot...
                            )
                i += 1

            else:
                fig = plt.figure(min_aspect_ratio=1, max_aspect_ratio=1)
                # First index is colour, second index is X

                # Note - putting negative does cool weird stuff
                bars = plt.bar(
                    scenario.ticks, values,
                    colors=colors,
                    # display_legend=False,
                    # labels=["Good Customers", "Bad Customers"],
                )
                siz = "4in"
                fig.layout.width = siz
                fig.layout.height = siz

                fig.title = title
                fig.axes[0].color = fig.axes[1].color = "Black"
                plt.ylabel(ylabel)
                plts.append(fig)

        if backend == "matplotlib":
            fig.savefig("images/" + figure_name + ".png", bbox_inches="tight", dpi=300)

        elif backend == "bqplot":
            box = widgets.HBox(plts)
            box.layout.width = "90%"
            display(box)

        figure_name = f"Profitability_{id}_{desc}"
# plot all the fairness metrics


def plot_fairness_metrics(scenarios, backend):
    for id in scenarios:
        # Make scenario data
        scenario = scenarios[id]
        desc = scenario.description

        train_data = scenario.gendata("train")
        test_data = scenario.gendata("test")
        np.random.seed(42)
        X_train, y_train = scenario.get_modelling_data(train_data)
        X_test, y_test = scenario.get_modelling_data(test_data)
        if "pinterest" in train_data.columns:
            ren = {
                "pinterest": "browser data",
            }
            # warning: train data seems to be writable in-place
            X_train = X_train.rename(columns=ren)
            X_test = X_test.rename(columns=ren)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        # X_train, y_train = scenario.get_modelling_data(train_data)
        # X_test, y_test = scenario.get_modelling_data(test_data)
        data = {
            'training': (X_train, y_train, train_data.disadv_flag),
            'deployment': (X_test, y_test, test_data.disadv_flag),
        }
        remap = {
            'training': ' (anticipated)',
            'deployment': '',
        }

        for cohort, (X, y, prot) in data.items():

            if (cohort == "training") and (id not in [4.]):
                continue

            y_pred_proba = clf.predict_proba(X)[:, 1] # todo fully we should have held-out set as well (train/test/deploy)
            y_pred = y_pred_proba.copy()
            y_pred[prot == 0] = y_pred[prot == 0] > scenario.decision_thresh[0]
            y_pred[prot == 1] = y_pred[prot == 1] > scenario.decision_thresh[1]
            # y_pred = clf.predict(X)  # todo fully we should have held-out set as well (train/test/deploy)
            print("Metrics for ", cohort)
            print(f"for Scenario {id}: {desc}")
            eo = metrics(id, y, y_pred, prot, remap[cohort],
                         x_ticks=scenario.ticks, pname=scenario.protected,
                         backend=backend
                         )


def make_fairness_maps(scenarios):
    for id in scenarios:
        # Make scenario data
        scenario = scenarios[id]
        desc = scenario.description
        print(desc)
        train_data = scenario.gendata("train")
        test_data = scenario.gendata("test")
        np.random.seed(42)
        X_train, y_train = scenario.get_modelling_data(train_data)
        X_test, y_test = scenario.get_modelling_data(test_data)
        if "pinterest" in train_data.columns:
            ren = {
                "pinterest": "browser data",
            }
            # warning: train data seems to be writable in-place
            X_train = X_train.rename(columns=ren)
            X_test = X_test.rename(columns=ren)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # we can't consider post processing mitigation
        # without a probabilistic classifier
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        y = y_test.astype(bool)
        A = test_data.disadv_flag.astype(bool)

        make_fairness_map(y_pred_prob, y, A, scenario.ticks, rank=True)
        figure_name = f"Map_{id}_{desc}"
        mplt.savefig("images/" + figure_name + ".png", bbox_inches="tight", dpi=300)


def make_fairness_map(y_pred_prob, y, A, ticks, rank):
    # Optionally convert to rank-scores
    if rank:
        # order = np.argsort(y_pred_prob)
        # y_pred_prob[order] = np.linspace(0., 1., len(y_pred_prob)+2)[1:-1]

        for a in [0, 1]:
            inds = np.where(A==a)[0]
            order = np.argsort(y_pred_prob[inds])
            y_pred_prob[inds[order]] = np.linspace(0., 1., len(order)+2)[1:-1]


    # Post-processing fairness

    # 1. Discretise Scores
    nbin=100
    scores = (y_pred_prob * nbin).astype(int)

    # Count scores in the percentile ranges
    # Positives

    P0 = np.bincount(scores[~A & y], minlength=nbin)
    P1 = np.bincount(scores[A & y], minlength=nbin)

    # Negatives
    N0 = np.bincount(scores[~A & ~y], minlength=nbin)
    N1 = np.bincount(scores[A & ~y], minlength=nbin)

    def cumsum(ys):
        # cumulative sum from the left
        # Pad to include both select all and select none
        return np.hstack((0, np.cumsum(ys).astype(int)))

    def cumsumr(ys):
        # cumulative sum from the right
        return cumsum(ys[::-1])[::-1]

    # Simulate all selections
    tp0 = cumsumr(P0)
    tp1 = cumsumr(P1)
    fp0 = cumsumr(N0)
    fp1 = cumsumr(N1)
    fn0 = cumsum(P0)
    fn1 = cumsum(P1)
    tn0 = cumsum(N0)
    tn1 = cumsum(N1)
    n = len(y)

    # Now compute some metrics on a grid
    acc = ((tp0 + tn0)[:, None] + (tp1 + tn1)[None, :])/n
    eps = 1e-8
    opp0 = (tp0 + eps) / (tp0 + fn0 + eps)
    opp1 = (tp1+eps) / (tp1 + fn1 + eps)
    eo = opp0[:, None] / opp1[None, :]
    dem0 = (tp0 + fp0 + eps) / (tp0 + fp0 + tn0 + fn0 + eps) 
    dem1 = (tp1 + fp1 + eps) / (tp1 + fp1 + tn1 + fn1 + eps) 
    dempar = dem0[:, None] / dem1[None, :]
    prec0 = (tp0 + eps) / (tp0 + fp0 + eps)
    prec1 = (tp1+eps) / (tp1 + fp1 + eps)
    pp = prec0[:, None] / prec1[None, :]
    alpha = 0.75

    if rank:
        # Fraction selected requires a flip from ranked threshold
        acc = acc[::-1, ::-1]
        eo = eo[::-1, ::-1]
        pp = pp[::-1, ::-1]
        dempar = dempar[::-1, ::-1]

    mplt.figure(dpi=150)
    acc = mplt.contour(acc, levels=np.linspace(acc.min(), acc.max()-1e-2, 11)[1:], colors='k', alpha=0.5)
    mplt.plot(0, 0, 'k', alpha=0.5, label="Accuracy")
    mplt.contourf(eo, levels=[0.95, 1.05], colors='g', alpha=alpha)
    mplt.plot(0, 0, 'gs', alpha=alpha, label="Equal Opportunity (±5%)") #, visible=False)
    mplt.contourf(dempar, levels=[0.95, 1.05], colors='b', alpha=alpha)
    mplt.plot(0, 0, 'bs', alpha=alpha, label="Demographic Parity (±5%)") #, visible=False)
    mplt.contourf(pp, levels=[0.95, 1.05], colors='m', alpha=alpha)
    mplt.plot(0, 0, 'ms', alpha=alpha, label="Precision Parity (±5%)") #, visible=False)
    # mplt.plot([0, 100], [0, 100], 'k:')

    mplt.axis('square')
    mplt.gca().clabel(acc, inline=1, fontsize=10)
    mplt.legend(bbox_to_anchor=(1.9, 0.5));
    mplt.title("Fairness Map")
    # mplt.xlabel(ticks[1] + " Cutoff")
    # mplt.ylabel(ticks[0] + " Cutoff");
    def lower(s):
        if s == "SE Asian":
            return s
        elif s == "Indigenous":
            return s
        else:
            # return s.lower()
            return s
    mplt.xlabel(f"Percentage of {lower(ticks[1])} applicants selected")
    mplt.ylabel(f"Percentage of {lower(ticks[0])} applicants selected");



if __name__ == "__main__":
    main()
