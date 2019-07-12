# Classification of Mental Health Risk Using Social Media Text 
![MH](https://bryantarchway.com/wp-content/uploads/2018/10/MentalHealth_Flickr.jpg)

# Motivation:

   Mental-Health clinicians face many obstacles, among them include the ability to diagnose patients with serious mental illness when typically they are only allotted sporadic one hour sessions in which to collect as much information as possible to create an accurate and potentially life-altering diagnosis. At the moment, the best supplements that clinicians have include surveys, out-dated questionnaires, and family interview to gather more information, which most patients understandably object to due to privacy concerns. Having access to additional data would indeed be beneficial to mental health clinicians in order to assist in accurately identifying patients who should be diagnosed with depression, anxiety, or most importantly, having suicidal ideation. 

   Upon searching the literature, one <a href="https://journals.sagepub.com/doi/full/10.1177/2167702617747074">article</a> had used NLP to identify common "absolutist" words (i.e. "everything", "always", "nothing", "all", etc...) that were found to be used more frequently in depressed, anxious, and suicidal people. In psychology and cognitive therapy, it is widely understood that absolutist thinking is a core dysfunction, specifically among those who are diagnosed with depression, anxiety and suicidal ideation. These absolutist words were identified as being used significantly more often in depression, anxiety, and suicide related forums when compared to other clinical and non-clinical forums. Suicide-related forums were also found to have a statistically significant increased usage of these words compared to depression and anxiety related forums. The 2018 study, which was done in the United Kingdom, used <a href="https://journals.sagepub.com/doi/suppl/10.1177/2167702617747074/suppl_file/Table_S1_Supplemental_Material.pdf">British forums</a> for their data analysis.

   Having access to additional data would indeed be beneficial to mental health clinicians in order to assist in accurately identifying patients who should be treated for depression, anxiety, or most importantly, suicidal ideation. If there is a way to incorporate NLP in new ways, combined with what we already know about the language of mental health patients, in order to enhance diagnostic data, clinicians and patients both stand to benefit immensely.


# Methods:

I first repeated the article's work by performing my own hypothesis test to confirm their results. I scraped data from similar forums found on reddit. The following table shows the characteristics of the forums that I scraped from for this project:

![Forum Characteristics](https://i.postimg.cc/K8Y9RJn1/Screen-Shot-2019-04-24-at-12-13-21-PM.png)

I then examined the posts throughout reddit belonging to these self-identifying mental health redditors, and not only in the mental health forums I originally retrieved the posts from. I compared their universal reddit posts with other authors' universal reddit posts whom had never written in a depression/suicidal/anxiety forum, in order to build a classifier that uses text to determine if a post is from an author who is depressed, anxious, or suicidal or not. Importantly, using the same authors' posts and not limiting to mental health related forums only helps to generalize the results.

I then built another classifier using deep learning to classify depressed and anxious redditors from those who are experiencing suicidal ideation. I utilized both LSTM and GRU Deep Neural Networks, as well as Count Vectorization and Term Frequency-Inverse Document Frequency Vectorization.

Finally, I built a third classifier, adding in calculable features such as post length, percentage of absolutist words used, sentiment, and subjectivity, as well as the time of day of the post. I was able to extract some important features from my classifiers for potential future research and academic purposes. With the features that I was able to extract, I created a <a href="https://suicidal-ideation-tracker-tool.herokuapp.com/">Dashboard</a> prototype for clinicians to input a patient's social media handle, and a time frame, in order to track statistics that may correlate with the presence of mental health issues overtime. 

![Snapshot](https://i.postimg.cc/XqwSyvF7/Screen-Shot-2019-04-24-at-9-54-29-PM.png)

# Results:

### Three Part Hypothesis Test yielded the following results:
<a href="https://github.com/lpilossoph/Capstone-Project/blob/master/Part%201%20-%20Hypothesis%20Testing.ipynb">Part 1 - Hypothesis Tesing - Link to Notebook</a>

1. There is a statistically significant increase in absolutist words used in mental health related forums, compared with hobby and leisure forums.

2. There is a statistically significant increase in absolutist words used in mental health related forums, compared with other non-mental health related chronic illness forums (excluding hobby/leisure forums).

3. There is a statistically significant increase in absolutist words used in forums specifically addressing suicidal ideation, compared with other mental health related forums.

![Suicide Forum Wordcloud](https://i.postimg.cc/zBBLZHWH/download.png)

### Mental Health Risk / Non-Mental Health Risk reddit post classifier yielded the following results:
<a href="https://github.com/lpilossoph/Capstone-Project/blob/master/Part%202%20-%20Mental%20Health%20Risk%20Classifier.ipynb">Part 2 - Mental Health Risk Classifier - Link to Notebook</a>

Multinomial Naive Bayes Results
* Accuracy:  0.77
* Precision:  0.76
* Recall:  0.78
* F1 Score:  0.77

![MNB Confusion Matrix1](https://i.postimg.cc/3RxHtgGV/download-3.png)


### Suicidal Ideation Risk/ Non-Suicidal Ideation Risk reddit post classifier yielded the following results:
<a href="https://github.com/lpilossoph/Capstone-Project/blob/master/Part%203%20-%20Suicide%20Risk%20Classifier.ipynb">Part 3 - Suicidal Ideation Risk Classifier - Link to Notebook</a>

Multinomial Naive Bayes Results
* Accuracy:  0.66
* Precision:  0.64
* Recall:  0.77
* F1 Score:  0.70

![MNB Confusion Matrix2](https://i.postimg.cc/DyP33jcH/download-4.png)

### Feature Extraction found the following features were most important:

![Feature Importance](https://i.postimg.cc/Lhd71Srt/Screen-Shot-2019-04-26-at-3-56-52-PM.png)

#### Words in post:
* "anxiety"
* "suicide"
* "die"
* "kill"
* "life"
* "anymore"

#### Features of post:
* Percentage Absolutist Words used - Higher percentage contributed to more suicidal posts
* Sentiment 
* Subjectivity

<a href="https://github.com/lpilossoph/Capstone-Project/blob/master/Part%204%20-%20Feature%20Extraction%20.ipynb">Part 4 - Feature Extraction and Interpretation - Link to Notebook</a>


# Use Cases:

The ability to classify specific mental health issues using text has many use cases for clinicians and those who love anyone suffering with mental illness. We all communicate through text on a daily basis, whether it be on our phones, in forums, or via email. If our language could assist in a clinical diagnosis of depression or suicidal ideation, it could potentially be life-saving. Social media participants could be screened for at risk users leading to *meaningful* targeted ads that could potentially help them. In the healthcare setting, a Clinical Dashboard similar to the prototype I <a href="https://suicidal-ideation-tracker-tool.herokuapp.com/">created</a> could be useful for providers in order to track how treatment with therapy and/or medications is affecting progress overtime. 

![grab-landing-page](https://s3.gifyu.com/images/screen.gif)
<a href="https://suicidal-ideation-tracker-tool.herokuapp.com/">
   
   ### Link to Dashboard</a>
   
# Limitations

Unfortunately, the lack of access to reliable and accurate data in this arena is a major limitation. Using redditors who had posted in suicide/depression/anxiety forums was the only way I could assume the presence of mental illness with the limited data available to me. The comparison forums provide another major limitation because there is no guarantee that these redditors are not also suffering from mental illness. 

Future work would involve similar classification building and deep learning using more accurate and reliably labeled data. If this project is any indication of how well a trained classifier and deep neural network can perform, it would be a major breakthrough to be able to work with actual clinical data!

### Blog Post Link
http://lpilossoph.github.io/reddit_scraping_with_praw
