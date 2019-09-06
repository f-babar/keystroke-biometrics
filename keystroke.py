from scipy.spatial.distance import euclidean, cityblock
import numpy as np
np.set_printoptions(suppress=True)
import pandas
from sklearn import metrics

class KeystrokeBiometrics:

    def __init__(self, subjects):
        self.user_scores = []
        self.user_signatures = []
        self.imposter_scores = []
        self.subjects = subjects
        self.reference_threshold = 0.85
        self.training_samples = 200
        self.users_group = {}

    def training(self):
        for subject in subjects:
            all_samples = data.loc[data.subject == subject, "H.period":"H.Return"]
            training_data = all_samples[:200]
            mean_vector = training_data.mean().values
            std_vector = training_data.std().values
            max_vector = training_data.max().values
            min_vector = training_data.min().values

            '''--------------------------------------
                Generate Signature for each user
            --------------------------------------'''

            signature = dict()
            signature["subject"] = subject
            signature["mean_vector"] = mean_vector
            signature["std_vector"] = std_vector
            signature["max_vector"] = max_vector
            signature["min_vector"] = min_vector

            self.user_signatures.append(signature)

    def testing(self, subjects):
        str1 = ''.join(str(e) for e in subjects)
        print("---------- TESTING OF ANOMALY DETECTOR FOR subjects [" + str1 + "] ----------")
        for subject in subjects:
            subject_data = data.loc[data.subject == subject, "H.period":"H.Return"]
            testing_data = subject_data[390:]
            result = self.anomaly_detector(subject, testing_data)
            match_count = result['match_count']
            print("Total tested samples:", testing_data.shape[0])
            print("Match Count:", match_count)

    def evaluation(self):
        EERs = []
        FARs = []
        FRRs = []
        for test_subject in subjects:
            subject_data = data.loc[data.subject == test_subject, "H.period":"H.Return"]
            testing_data = subject_data[:200]
            result = self.anomaly_detector(test_subject, testing_data)
            match_count = result['match_count']
            users_score = result['scores']
            '''--------------------------------------
                  Testing with Imposters data
            --------------------------------------'''
            imposter_data = data.loc[data.subject != test_subject, :]
            testing_data = imposter_data.groupby("subject").head(20).loc[:, "H.period":"H.Return"]
            result = self.anomaly_detector(test_subject, testing_data)
            imposters_score = result['scores']
            match_count = result['match_count']
            evaluation = self.calculateEER(users_score, imposters_score)
            FARs.append(evaluation['FAR'])
            FRRs.append(evaluation['FRR'])
            EERs.append(evaluation['EER'])

        print("EER:", round(np.mean(EERs), 2))
        print("FAR:", round(np.mean(FARs), 2))
        print("FRR:", round(np.mean(FRRs), 2))

    def identifyUserGroups(self):
        sheeps = []
        goats = []
        lambs = []
        wolves = []

        for test_subject in subjects:
            subject_data = data.loc[data.subject == test_subject, "H.period":"H.Return"]
            real_testing_data = subject_data[0:100]
            result = self.anomaly_detector(test_subject, real_testing_data)
            real_match_count = result['match_count']
            real_users_score = result['scores']

            '''--------------------------------------
                  Testing with Imposters data
            --------------------------------------'''
            imposter_data = data.loc[data.subject != test_subject, :]
            testing_data = imposter_data.groupby("subject").head(10).loc[:, "subject":"H.Return"]

            result = self.anomaly_detector(test_subject, testing_data, 'imposters')
            imposters_score = result['scores']
            match_count = result['match_count']

            unique_elements, counts_elements = np.unique(result["imposter_match"], return_counts=True)

            for j in range( len(unique_elements)):
                if (counts_elements[j] / 10) >= 0.85:
                    wolves.append(unique_elements[j])

            percentage = real_match_count / real_testing_data.shape[0]
            if percentage >= 0.7:
                sheeps.append(test_subject)
            else:
                goats.append(test_subject)

            evaluation = self.calculateEER(real_users_score, imposters_score)
            if(evaluation['FRR'] >= 0.7):
                goats.append(test_subject)

            if evaluation['FAR'] >= 0.85:
                lambs.append(test_subject)

            percentage = match_count / testing_data.shape[0]
            if percentage >= 0.85:
                lambs.append(test_subject)

        unique_elements, counts_elements  = np.unique(wolves, return_counts=True)
        wolves = []
        for j in range(len(unique_elements)):
            if (counts_elements[j] / 50) >= 0.5:
                wolves.append(unique_elements[j])

        print("Sheep:", sheeps)
        print("Goats:", goats)
        print("lambs:", lambs)
        print("Wolves:", wolves)

    def calculateEER(self, user_scores, imposter_scores):
        labels = [0] * len(user_scores) + [1] * len(imposter_scores)
        fpr, tpr, thresholds = metrics.roc_curve(labels, user_scores + imposter_scores)
        FRR = 1 - tpr
        FAR = fpr
        dists = FRR - FAR
        idx1 = np.argmin(dists[dists >= 0])
        idx2 = np.argmax(dists[dists < 0])
        x = [FRR[idx1], FAR[idx1]]
        y = [FRR[idx2], FAR[idx2]]
        a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
        eer = x[0] + a * (y[0] - x[0])
        result = dict()
        result['FAR'] = np.mean(FAR)
        result['FRR'] = np.mean(FRR)
        result['EER'] = eer
        return result

    def anomaly_detector(self, subject, testing_data, test_users = ''):
        user_signature = list(filter(lambda user: user['subject'] == subject, self.user_signatures))
        if len(user_signature) is 0:
            return "User signature doesn't exit in the system."
        result = {}
        result["scores"] = []
        result["match_count"] = 0
        result["not_match_count"] = 0
        result["distance"] = []
        result["subject"] = []
        result["imposter_match"] = []
        for i in range(testing_data.shape[0]):

            if test_users == "imposters":
                imposter_subject = testing_data.iloc[i].values[0]
                imposter_testing_data = testing_data.loc[:, "H.period":"H.Return"]
                test_vector = imposter_testing_data.iloc[i].values
            else:
                test_vector = testing_data.iloc[i].values
            mean_vector = user_signature[0]["mean_vector"]
            max_vector = user_signature[0]["max_vector"]

            distance = euclidean(test_vector, mean_vector)
            result["scores"].append(distance)
            max_distance = euclidean(mean_vector, max_vector)
            beta = distance / max_distance
            distance = 1 / (1 + (beta * distance))

            result["distance"].append(distance)
            if distance >= 0 and distance < self.reference_threshold:
                result["not_match_count"] += 1
            elif distance >= self.reference_threshold and distance <= 1:
                result["match_count"] += 1
                result["subject"].append(subject)
                if test_users == 'imposters':
                    result["imposter_match"].append(imposter_subject)
            else:
                result["not_match_count"] += 1

        return result

if __name__ == "__main__":

    path = "dataset/DSL-StrongPasswordData.csv"
    data = pandas.read_csv(path)
    subjects = data["subject"].unique()
    kb_obj = KeystrokeBiometrics(subjects)

    '''---------- Training ---------'''
    kb_obj.training()

    '''---------- Testing of Anomaly Detector ---------'''
    kb_obj.testing(['s002'])

    print("----------EVALUATION OF ANOMALY DETECTOR----------")
    kb_obj.evaluation()

    print("----------IDENTIFY USER GROUPS (Sheeps, Goats, Lambs, Wolves)----------")
    kb_obj.identifyUserGroups()

