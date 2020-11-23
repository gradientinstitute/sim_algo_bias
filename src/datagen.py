import numpy as np
import pandas as pd


class Params(object):
    """Hold the parameters for a given dataset."""

    def __init__(self):
        self.n_adv = None
        self.n_dis = None
        self.p_mat = None
        self.mat_scale = None
        self.sal_mu_adv = None
        self.sal_mu_dis = None
        self.p_pinterest_adv = None
        self.p_pinterest_dis = None
        self.dist_sig_adv = None
        self.dist_sig_dis = None
        self.dist_scale = None
        self.label_bias = 0
        self.thresh = None
        self.noise = None

    def set_population_counts(self, n_advantaged, n_disadvantaged):
        self.n_adv = n_advantaged
        self.n_dis = n_disadvantaged

    def set_leave_parameters(self, prob_leave, leave_scale):
        self.p_mat = prob_leave
        self.mat_scale = leave_scale

    def set_salary_parameters(self, mu_advantaged, mu_disadvantaged):
        self.sal_mu_adv = mu_advantaged
        self.sal_mu_dis = mu_disadvantaged

    def set_pinterest_parameters(self, prob_pinterest_adv, prob_pinterest_dis):
        self.p_pinterest_adv = prob_pinterest_adv
        self.p_pinterest_dis = prob_pinterest_dis

    def set_dist_parameters(self, dist_sig_adv, dist_sig_dis, dist_scale):
        self.dist_sig_adv = dist_sig_adv
        self.dist_sig_dis = dist_sig_dis
        self.dist_scale = dist_scale

    def set_label_bias(self, label_bias):
        self.label_bias = label_bias

    def set_threshold(self, threshold):
        self.thresh = threshold

    def set_noise(self, noise):
        self.noise = noise

    def is_valid(self):
        return False


class Scenario(object):

    def __init__(self, scenario_id, description=""):
        self.scenario_id = scenario_id
        self.description = description
        self.datasets = {"train": Params(), "test": Params()}
        self.ticks = ["Male", "Female"]  # for plots
        self.protected = "women"
        self.proxy_name = "browser data"
        self.decision_thresh = [0.5, 0.5]
        # self.groups = ["men", "women"]  # for stakeholders

    def set_switches(self, drop_attrib, drop_proxy, covariate_shift):
        self.drop_attrib = drop_attrib  # Setting this to false means women are penalised for their gender's historic performance
        self.drop_proxy = drop_proxy
        self.cov_shift = covariate_shift

    def set_population_counts(self, dataset, n_advantaged, n_disadvantaged):
        if dataset == "all":
            for param in self.datasets.values():
                param.set_population_counts(n_advantaged, n_disadvantaged)
        else:
            self.datasets[dataset].set_population_counts(n_advantaged, n_disadvantaged)

    def set_leave_parameters(self, dataset, prob_leave, leave_scale):
        if dataset == "all":
            for param in self.datasets.values():
                param.set_leave_parameters(prob_leave, leave_scale)
        else:
            self.datasets[dataset].set_leave_parameters(prob_leave, leave_scale)

    def set_salary_parameters(self, dataset, mu_advantaged, mu_disadvantaged):
        if dataset == "all":
            for param in self.datasets.values():
                param.set_salary_parameters(mu_advantaged, mu_disadvantaged)
        else:
            self.datasets[dataset].set_salary_parameters(mu_advantaged, mu_disadvantaged)

    def set_pinterest_parameters(self, dataset, prob_pinterest_adv, prob_pinterest_dis):
        if dataset == "all":
            for param in self.datasets.values():
                param.set_pinterest_parameters(prob_pinterest_adv, prob_pinterest_dis)
        else:
            self.datasets[dataset].set_pinterest_parameters(prob_pinterest_adv, prob_pinterest_dis)

    def set_dist_parameters(self, dataset, dist_sig_adv, dist_sig_dis, dist_scale):
        if dataset == "all":
            for param in self.datasets.values():
                param.set_dist_parameters(dist_sig_adv, dist_sig_dis, dist_scale)
        else:
            self.datasets[dataset].set_dist_parameters(dist_sig_adv, dist_sig_dis, dist_scale)

    def set_label_bias(self, dataset, label_bias):
        if dataset == "all":
            for param in self.datasets.values():
                param.set_label_bias(label_bias)
        else:
            self.datasets[dataset].set_label_bias(label_bias)

    def set_threshold(self, dataset, threshold):
        if dataset == "all":
            for param in self.datasets.values():
                param.set_threshold(threshold)
        else:
            self.datasets[dataset].set_threshold(threshold)

    def set_noise(self, dataset, noise):
        if dataset == "all":
            for param in self.datasets.values():
                param.set_noise(noise)
        else:
            self.datasets[dataset].set_noise(noise)

    def gendata(self, dataset):
        params = self.datasets[dataset]

        # Maternity leave flag
        mat_leave_adv = np.zeros(params.n_adv)
        mat_leave_dis = np.random.binomial(1, params.p_mat, params.n_dis)

        # Salary flag
        salary_adv = (np.random.randn(params.n_adv)) + params.sal_mu_adv
        salary_dis = (np.random.randn(params.n_dis)) + params.sal_mu_dis

        # Flag for visiting pinterest
        pint_adv = np.random.binomial(1, params.p_pinterest_adv, params.n_adv)
        pint_dis = np.random.binomial(1, params.p_pinterest_dis, params.n_dis)

        # Distance from an Energy Node
        dist_adv = np.abs((np.random.randn(params.n_adv) * params.dist_sig_adv) + 0)
        dist_dis = np.abs((np.random.randn(params.n_dis) * params.dist_sig_dis) + 0)

        mat_leave = np.hstack((mat_leave_adv, mat_leave_dis))
        salary = np.hstack((salary_adv, salary_dis))
        distance = np.hstack((dist_adv, dist_dis))
        pint = np.hstack((pint_adv, pint_dis))

        # Flag for disadvantaged group
        disadv_flag = np.hstack((np.zeros(params.n_adv), np.ones(params.n_dis)))

        # Add noise for unobserved variables
        noise = np.random.randn(len(disadv_flag)) * params.noise

        # Customer Goodness:

        if self.cov_shift:
            goodness = np.zeros(len(disadv_flag))

            goodness[disadv_flag == 0] = salary[disadv_flag == 0] -\
                                (params.dist_scale * distance[disadv_flag == 0]) -\
                                (params.mat_scale * mat_leave[disadv_flag == 0]) -\
                                params.label_bias * disadv_flag[disadv_flag == 0] +\
                                noise[disadv_flag == 0]

            # different penalty for dist_scale
            goodness[disadv_flag == 1] = salary[disadv_flag == 1] -\
                                (0.2 * params.dist_scale * distance[disadv_flag == 1]) -\
                                (params.mat_scale * mat_leave[disadv_flag == 1]) -\
                                params.label_bias * disadv_flag[disadv_flag == 1] +\
                                noise[disadv_flag == 1]

        else:
            goodness = salary - (params.dist_scale * distance) - (params.mat_scale * mat_leave) - params.label_bias * disadv_flag + noise

        good = goodness > params.thresh

        data = pd.DataFrame(np.array(np.vstack((salary, distance, pint, disadv_flag, good))).T, columns=["income", "cost", "pinterest", "disadv_flag", "good"])
        data = data.sample(frac=1)

        return data

    def get_modelling_data(self, data):
        drop_cols = ["good", "cost"]
        if self.drop_attrib:
            drop_cols.append("disadv_flag")
        if self.drop_proxy:
            drop_cols.append("pinterest")

        X = data.drop(columns=drop_cols)
        y = data["good"].values
        return X, y




def make_scenarios():
    """An exact copy of the notebook for now."""
    scenarios = {}

    # Baseline
    scenario_id = 0
    scenario = Scenario(scenario_id, "Baseline")
    scenario.set_switches(False, False, False)                    # Proxies are included and so is gender
    scenario.set_population_counts("train", 10000, 10000)
    scenario.set_population_counts("test", 100000, 100000)
    scenario.set_leave_parameters("all", 0., 1.5)        # No difference in extended leave
    scenario.set_salary_parameters("all", 0., 0.)         # No difference in salary
    scenario.set_pinterest_parameters("all", 0.2, 0.8)  # Difference in pinterest but should be irrelevant
    scenario.set_dist_parameters("all", 0.6, 0.6, 0.3)
    scenario.set_label_bias("all", 0)
    scenario.set_threshold("all", -0.5)
    scenario.decision_thresh = [0.5, 0.5]
    scenario.set_noise("all", 0.2)
    scenarios[scenario_id] = scenario

    # # Different Base Rates
    # scenario_id = 10
    # scenario = Scenario(scenario_id, "Different base rates")
    # scenario.set_switches(False, False, False)
    # scenario.set_population_counts("train", 10000, 10000)
    # scenario.set_population_counts("test", 100000, 100000)
    # scenario.set_leave_parameters("all", 0.6, 0.6)    # Women are less good customers because of the extra leave
    # scenario.set_salary_parameters("all", 0.2, -0.2)    # Women earn a lower *base* salary on average
    # scenario.set_pinterest_parameters("all", 0.2, 0.8)
    # scenario.set_dist_parameters("all", 0.6, 0.6, 0.3)
    # scenario.set_label_bias("all", 0)
    # scenario.set_threshold("all", -0.5)
    # scenario.set_noise("all", 0.2)
    # scenarios[scenario_id] = scenario

    # # Different base rate with proxy feature
    # scenario_id = 15
    # scenario = Scenario(scenario_id, "Different base rates with biased proxy")
    # scenario.set_switches(True, False, False)
    # scenario.set_population_counts("train", 10000, 10000)
    # scenario.set_population_counts("test", 10000, 10000)
    # scenario.set_leave_parameters("all", 0.6, 0.6)    # Women are less good customers because of the extra leave
    # scenario.set_salary_parameters("all", 0.2, -0.2)    # Women earn a lower *base* salary on average
    # scenario.set_pinterest_parameters("all", 0.2, 0.8)
    # scenario.set_dist_parameters("all", 0.6, 0.6, 0.3)
    # scenario.set_label_bias("all", 0)
    # scenario.set_threshold("all", -0.5)
    # scenario.set_noise("all", 0.2)
    # scenarios[scenario_id] = scenario


    # Different Base Rates - Indigenous
    scenario_id = 10
    scenario = Scenario(scenario_id, "Different base rates")
    scenario.set_switches(True, True, False)
    scenario.set_population_counts("train", 10000, 10000)
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("all", 0., 0.)
    scenario.set_salary_parameters("all", 0.3, -0.3)
    scenario.set_pinterest_parameters("all", 0.5, 0.5)
    scenario.set_dist_parameters("all", 0.3, 0.9, 0.3)
    scenario.set_label_bias("all", 0.)     # Discrimination against the disadvantaged class
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 0.4)
    scenario.ticks = ["Non- \n Indigenous", "Indigenous"]  # for plots
    scenario.proxy_name="zip code"
    scenario.decision_thresh = [0.5, 0.5]
    scenario.protected = "Indigenous"
    scenarios[scenario_id] = scenario


    # Different Base Rates - Indigenous
    scenario_id = 15
    scenario = Scenario(scenario_id, "Different base rates post process ")
    scenario.set_switches(True, True, False)
    scenario.set_population_counts("train", 10000, 10000)
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("all", 0., 0.)
    scenario.set_salary_parameters("all", 0.3, -0.3)
    scenario.set_pinterest_parameters("all", 0.5, 0.5)
    scenario.set_dist_parameters("all", 0.3, 0.9, 0.3)
    scenario.set_label_bias("all", 0.)     # Discrimination against the disadvantaged class
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 0.4)
    scenario.ticks = ["Non- \n Indigenous", "Indigenous"]  # for plots
    scenario.proxy_name="zip code"
    scenario.decision_thresh = [0.5, 0.1]
    scenario.protected = "Indigenous"
    scenarios[scenario_id] = scenario


    # Historical Bias with gender feature
    scenario_id = 20
    scenario = Scenario(scenario_id, "Historical Bias")
    scenario.set_switches(False, False, False) 
    scenario.set_population_counts("train", 10000, 10000)
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("train", 0.6, 0.5)
    scenario.set_leave_parameters("test", 0.6, 0.)         # The penalty of leave has reduced over time
    scenario.set_salary_parameters("all", 0., 0.)
    scenario.set_pinterest_parameters("all", 0.1, 0.9)
    scenario.set_dist_parameters("all", 0.6, 0.6, 0.3)
    scenario.set_label_bias("all", 0)
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 0.2)
    scenarios[scenario_id] = scenario

    # Historical Bias with gender feature
    scenario_id = 22
    scenario = Scenario(scenario_id, "Historical Bias - Remove Protected")
    scenario.set_switches(True, False, False) 
    scenario.set_population_counts("train", 10000, 10000)
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("train", 0.6, 0.5)
    scenario.set_leave_parameters("test", 0.6, 0.0)         # The penalty of leave has reduced over time
    scenario.set_salary_parameters("all", 0., 0.)
    scenario.set_pinterest_parameters("all", 0.1, 0.9)
    scenario.set_dist_parameters("all", 0.6, 0.6, 0.3)
    scenario.set_label_bias("all", 0)
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 0.2)
    scenarios[scenario_id] = scenario

    # Training data is more recent. Gender feature but small inherent difference between genders (extended leave)
    scenario_id = 25  # hmm beware float keys
    scenario = Scenario(scenario_id, "Historical Bias mitigated via updated Training data")
    scenario.set_switches(False, False, False) 
    scenario.set_population_counts("train", 10000, 10000)
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("all", 0.6, 0.)         # The penalty of leave has reduced over time
    scenario.set_salary_parameters("all", 0., 0.)
    scenario.set_pinterest_parameters("all", 0.1, 0.9)
    scenario.set_dist_parameters("all", 0.6, 0.6, 0.3)
    scenario.set_label_bias("all", 0)
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 0.2)
    scenarios[scenario_id] = scenario



    # Biased label in training data but fixed in deployment data
    scenario_id = 30
    scenario = Scenario(scenario_id, "Biased labels in training (but not deployment)")
    scenario.set_switches(False, True, False)
    scenario.set_population_counts("train", 10000, 10000)
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("all", 0., 0.)
    scenario.set_salary_parameters("all", 0., 0.)
    scenario.set_pinterest_parameters("all", 0.2, 0.8)
    scenario.set_dist_parameters("all", 0.6, 0.6, 0.3)
    scenario.set_label_bias("all", 0.2)     # Discrimination against the disadvantaged class
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 0.2)
    scenario.ticks = ["Gen pop", "SE Asian"]  # for plots
    # scenario.groups = ["non Chinese", "Chinese"]  # for stakeholders
    scenario.protected = "SE Asian"
    scenarios[scenario_id] = scenario

    scenario_id = 35
    # Biased label in training data and deployment data
    scenario = Scenario(scenario_id, "Biased Labels in both training and deployment")
    scenario.set_switches(False, True, False)
    scenario.set_population_counts("train", 10000, 10000)
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("all", 0., 0.)
    scenario.set_salary_parameters("all", 0., 0.)
    scenario.set_pinterest_parameters("all", 0.2, 0.8)
    scenario.set_dist_parameters("all", 0.6, 0.6, 0.3)
    scenario.set_label_bias("all", 0.)     # Discrimination against the disadvantaged class
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 0.2)
    scenario.ticks = ["Gen pop", "SE Asian"]  # for plots
    # scenario.groups = ["non Chinese", "Chinese"]  # for stakeholders
    scenario.protected = "SE Asian"
    scenarios[scenario_id] = scenario

    # Under-representation
    scenario_id = 41
    scenario = Scenario(scenario_id, "Under-representation")
    scenario.set_switches(False, True, True)            # drop true attr, drop proxy, Covariate shift in distance
    scenario.set_population_counts("train", 10000, 10) # Disadvantaged community underrepresented in training data
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("all", 0., 0.)
    scenario.set_salary_parameters("all", .35, 0.)
    scenario.set_pinterest_parameters("all", 0.2, 0.8)
    scenario.set_dist_parameters("all", 0.6, 0.1, 1.3)
    scenario.set_label_bias("train", 0)
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 1.4)
    scenario.ticks = ["25 & Over", "Under 25"]  # for plots
    # scenario.groups = ["---", "youths"]  # for stakeholders
    scenario.protected = "youths"
    scenarios[scenario_id] = scenario

    # Under-representation mitigated
    scenario_id = 415
    scenario = Scenario(scenario_id, "Under-representation (mitigated)")
    scenario.set_switches(False, True, True)             # Covariate shift
    scenario.set_population_counts("train", 10000, 10000) # Disadvantaged community underrepresented in training data
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("all", 0., 0.)
    scenario.set_salary_parameters("all", .35, 0.)
    scenario.set_pinterest_parameters("all", 0.2, 0.8)
    scenario.set_dist_parameters("all", 0.6, 0.1, 1.3)
    scenario.set_label_bias("train", 0)
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 1.4)
    scenario.ticks = ["25 & Over", "Under 25"]  # for plots
    scenario.protected = "youths"
    # scenario.groups = ["---", "youths"]  # for stakeholders
    scenarios[scenario_id] = scenario

    # Inflexible model
    scenario_id = 420
    scenario = Scenario(scenario_id, "Inflexible")
    scenario.set_switches(True, True, True)             # Covariate shift
    scenario.set_population_counts("train", 10000, 10000) # Disadvantaged community underrepresented in training data
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("all", 0., 0.)
    scenario.set_salary_parameters("all", .45, 0.)
    scenario.set_pinterest_parameters("all", 0.2, 0.8)
    scenario.set_dist_parameters("all", 0.6, 0.6, 1.3)
    scenario.set_label_bias("train", 0)
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 0.2)
    scenario.ticks = ["25 & Over", "Under 25"]  # for plots
    scenario.protected = "youths"
    # scenario.groups = ["---", "youths"]  # for stakeholders
    scenarios[scenario_id] = scenario

    scenario_id = 425
    scenario = Scenario(scenario_id, "Inflexible - fixed")
    scenario.set_switches(False, True, True)             # Covariate shift
    scenario.set_population_counts("train", 10000, 10000) # Disadvantaged community underrepresented in training data
    scenario.set_population_counts("test", 10000, 10000)
    scenario.set_leave_parameters("all", 0., 0.)
    scenario.set_salary_parameters("all", .45, 0.)
    scenario.set_pinterest_parameters("all", 0.2, 0.8)
    scenario.set_dist_parameters("all", 0.6, 0.6, 1.3)
    scenario.set_label_bias("train", 0)
    scenario.set_threshold("all", -0.7)
    scenario.set_noise("all", 0.2)
    scenario.ticks = ["25 & Over", "Under 25"]  # for plots
    scenario.protected = "youths"
    # scenario.groups = ["---", "youths"]  # for stakeholders
    scenarios[scenario_id] = scenario

    return scenarios


scenarios = make_scenarios()
