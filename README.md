# Categorizing Mental Health Risk in Social Media Posts 
![MH](MentalHealth_Flickr.jpg)
### Classifying mental health patients' posts and classifying suicidal ideation in posts using deep learning

# Motivation

   Mental-Health clinicians face many obstacles, among them include the ability to diagnose patients with serious mental illness when typically they are only allotted sporadic one hour sessions in which to collect as much information as possible to create an accurate and potentially life-altering diagnosis for the patient. At the moment, the best supplement that clinicians have include interviewing family and friends to gather more information, which most patients understandably object to due to privacy concerns.
Having access to additional data would indeed be beneficial to mental health clinicians to assist in accurately identifying patients who should be diagnosed with depression, anxiety, or most importantly, having suicidal ideation. 

Upon searching the literature, one <a href="https://journals.sagepub.com/doi/full/10.1177/2167702617747074">article</a> had used NLP to identify common "absolutist" words (i.e. "everything", "always", "nothing", "all", etc...) that were found to be used more frequently in depressed, anxious, and suicidal people. In psychology and cognitive therapy, it is widely understood that absolutist thinking is a core dysfunction in mental health patients, specifically among those who are diagnosed with depression, anxiety and suicidal ideation. These absolutist words were identified as being used significantly more often in depression, anxiety, and suicide related forums when compared to other clinical and non-clinical forums. Suicide-related forums were also found to have a statistically significant increased usage of these words compared to depression and anxiety related forums. The 2018 study, which was done in the United Kingdom, used <a href="https://journals.sagepub.com/doi/suppl/10.1177/2167702617747074/suppl_file/Table_S1_Supplemental_Material.pdf">British forums</a> for data analysis.

Having access to additional data other than patient interviews would indeed be beneficial to mental health clinicians in order to assist in accurately identifying patients who should be diagnosed with depression, anxiety, or most importantly, experiencing suicidal ideation. If there is a way to incorporate NLP in new ways, combined with what we already know about the language of mental health patients in order to enhance diagnostic data, clinicians and patients both stand to benefit greatly.


# The Plan

I plan to first repeat the article's work by performing my own hypothesis test to confirm their theory. I will scrape data from similar forums found on reddit.

I then plan to examine the posts throughout reddit of self-identifying mental health redditors. I will compares their posts with other authors not self-identifying as depressed/suicidal/anxious, in order to build a classifier that uses text to determine if a post is from an author who is depressed, anxious, or suicidal.

Next, I plan to build another classifier using deep learning to classify depressed and anxious redditors from those who are experiencing suicidal ideation. 

Finally, I will attempt to extract some features from my classifiers for potential future research and academic purposes. 

# Results

### Three Part Hypothesis Test yielded the following results
1. There is a statistically significant increase in absolutist words used in mental health related forums, compared with hobby and leisure forums

2. There is a statistically significant increase in absolutist words used in mental health related forums, compared with other non-mental health related chronic illness forums

3. There is a statistically significant increase in absolutist words used in forums specifically addressing suicidal ideation, compared with other mental health related forums

### Mental Health Risk / Non-Mental Health Risk reddit post classifier yielded the following results:

Multinomial Naive Bayes Results

Precision:  0.76
Recall:  0.78
Accuracy:  0.77
F1 Score:  0.77

### Suicidal Ideation Risk/ Non-Suicidal Ideation Risk reddit post classifier yielded the following results:

Multinomial Naive Bayes Results

Accuracy:  0.66
Precision:  0.64
Recall:  0.77
F1 Score:  0.70

### Feature Extraction found the following features were most important:




# Usage and Limitations

The ability to classify specific mental health issues using text has many use cases for clinicians and those who love anyone suffering with mental illness. We all communicate through text on a daily basis, whether it be on our phones, in forums, or via email. If our language could assist a clinician in a diagnosis of depression or suicidal ideation, it could potentially be life-saving. Unfortunately, the lack of accurate and meticulous data in this area a major limitation. Using redditors who had posted in suicide/depression/anxiety forums was the only way I could assume the presence of mental illness. The comparison forums however provide another major limitation because there is no guarantee that these redditors are not also suffering from mental illness. 

Future work would involve similar classification building and deep learning using more accurate data. If this project is any indication of how well a trained classifier and deep neural network can perform, it would be a major breakthrough to be able to work with reliable data!

# Blog Post Link
http://lpilossoph.github.io/reddit_scraping_with_praw
