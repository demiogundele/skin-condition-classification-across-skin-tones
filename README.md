# CSC211-course-project
Project Frame: Classify skin condition images (Eczema, Insect Bite, Folliculitis) across skin tones, using the SCIN dataset  (algorithmic, deep learning, LLM, and human baseline approaches), exploring the societal issue of racial bias in medical AI.

•	Framed a supervised learning image-classification problem (distinguishing between Eczema, Insect Bite, and Folliculitis) using Google’s SCIN dermatology dataset, which is designed to include a range of skin tones; filtered 516 confidence-weighted cases and built a stratified 336:180 training to test set split (seed 42) to prevent data leakage 
•	Developed a computer-vision algorithm in Python (OpenCV, scikit-image, NumPy) with YCbCr skin segmentation, Hough circle detection, contour analysis, and LBP/GLCM texture features feeding a rule-based classifier 


Citations:

Data: SCIN Dataset, Google Research 
https://github.com/google-research-datasets/scin
The SCIN Dataset is licensed with SCIN Data Use License
Modified: Filtered to Eczema, Insect Bite, and Folliculitis cases

Research Paper:


@article{10.1001/jamanetworkopen.2024.46615,
    author = {Ward, Abbi and Li, Jimmy and Wang, Julie and Lakshminarasimhan, Sriram and Carrick, Ashley and Campana, Bilson and Hartford, Jay and Sreenivasaiah, Pradeep K. and Tiyasirisokchai, Tiya and Virmani, Sunny and Wong, Renee and Matias, Yossi and Corrado, Greg S. and Webster, Dale R. and Smith, Margaret Ann and Siegel, Dawn and Lin, Steven and Ko, Justin and Karthikesalingam, Alan and Semturs, Christopher and Rao, Pooja},
    title = {Creating an Empirical Dermatology Dataset Through Crowdsourcing With Web Search Advertisements},
    journal = {JAMA Network Open},
    volume = {7},
    number = {11},
    pages = {e2446615-e2446615},
    year = {2024},
    month = {11},
    abstract = {Health datasets from clinical sources do not reflect the breadth and diversity of disease, impacting research, medical education, and artificial intelligence tool development. Assessments of novel crowdsourcing methods to create health datasets are needed.To evaluate if web search advertisements (ads) are effective at creating a diverse and representative dermatology image dataset.This prospective observational survey study, conducted from March to November 2023, used Google Search ads to invite internet users in the US to contribute images of dermatology conditions with demographic and symptom information to the Skin Condition Image Network (SCIN) open access dataset. Ads were displayed against dermatology-related search queries on mobile devices, inviting contributions from adults after a digital informed consent process. Contributions were filtered for image safety and measures were taken to protect privacy. Data analysis occurred January to February 2024.Dermatologist condition labels as well as estimated Fitzpatrick Skin Type (eFST) and estimated Monk Skin Tone (eMST) labels.The primary metrics of interest were the number, quality, demographic diversity, and distribution of clinical conditions in the crowdsourced contributions. Spearman rank order correlation was used for all correlation analyses, and the χ2 test was used to analyze differences between SCIN contributor demographics and the US census.In total, 5749 submissions were received, with a median of 22 (14-30) per day. Of these, 5631 (97.9\%) were genuine images of dermatological conditions. Among contributors with self-reported demographic information, female contributors (1732 of 2596 contributors [66.7\%]) and younger contributors (1329 of 2556 contributors [52.0\%] aged \&lt;40 years) had a higher representation in the dataset compared with the US population. Of 2614 contributors who reported race and ethnicity, 852 (32.6\%) reported a racial or ethnic identity other than White. Dermatologist confidence in assigning a differential diagnosis increased with the number of self-reported demographic and skin-condition–related variables (Spearman R = 0.1537; P \&lt; .001). Of 4019 contributions reporting duration since onset, 2170 (54.0\%) reported onset within less than 7 days of submission. Of the 2835 contributions that could be assigned a dermatological differential diagnosis, 2523 (89.0\%) were allergic, infectious, or inflammatory conditions. eFST and eMST distributions reflected the geographical origin of the dataset.The findings of this survey study suggest that search ads are effective at crowdsourcing dermatology images and could therefore be a useful method to create health datasets. The SCIN dataset bridges important gaps in the availability of images of common, short-duration skin conditions.},
    issn = {2574-3805},
    doi = {10.1001/jamanetworkopen.2024.46615},
    url = {https://doi.org/10.1001/jamanetworkopen.2024.46615},
    eprint = {https://jamanetwork.com/journals/jamanetworkopen/articlepdf/2826506/ward\_2024\_oi\_241322\_1731448364.35384.pdf},
}


