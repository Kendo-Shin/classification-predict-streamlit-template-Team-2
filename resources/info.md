###  How it works

This app requires the user to input text
(ideally a tweet relating to climate change), and will
classify it according to whether or not they believe in
climate change. Below you will find information about the data source
and a brief data description. Click on the checkbox below to `show the raw data`. You can have a look at more information
on the data used in training our models on the ``EDA`` page. Learn more about the prediction models available on the ``Model Information`` page, and make your choice of model to use on the
``Prediction`` page that you can navigate to in the sidebar. To learn more about ```DataTech```, the brain behind the app, check out the ``About Us`` page.

###  Data description as per source

The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo.

This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).

Each tweet is labelled as one of the following classes:

- 2 (News): the tweet links to factual news about climate change
- 1 (Pro): the tweet supports the belief of man-made climate change
- 0 (Neutral): the tweet neither supports nor refutes the belief of man-made climate change
- -1 (Anti): the tweet does not believe in man-made climate change
