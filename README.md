<h1 dir="auto"><a id="user-content-fifa23-ml-project-" class="anchor" aria-hidden="true" href="#fifa23-ml-project-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>FIFA23 ML Project <a target="_blank" rel="noopener noreferrer nofollow" href="https://camo.githubusercontent.com/d682520733daeec02595dc0e1252daee3dc4a3485f1feb6d2c05b772bbfab05f/68747470733a2f2f696d672e69636f6e73382e636f6d2f636f6c6f722f34382f6e756c6c2f666f6f7462616c6c2d7465616d2e706e67"><img src="https://camo.githubusercontent.com/d682520733daeec02595dc0e1252daee3dc4a3485f1feb6d2c05b772bbfab05f/68747470733a2f2f696d672e69636f6e73382e636f6d2f636f6c6f722f34382f6e756c6c2f666f6f7462616c6c2d7465616d2e706e67" data-canonical-src="https://img.icons8.com/color/48/null/football-team.png" style="max-width: 100%;"></a></h1>
<p dir="auto">Worked on this project during the AI training at Samsung Innovation Campus.</p>
<p dir="auto">Kaggle Notebook --&gt; <a href="https://www.kaggle.com/code/miraehab/eda-position-prediction-players-clustring" rel="nofollow">My Notebook</a><br>
Project Presentation --&gt; <a href="https://github.com/miraehab/FIFA-23-ML-Project/blob/main/presentation.pptx">Presentation</a></p>
<h2 dir="auto"><a id="user-content-table-of-contents-" class="anchor" aria-hidden="true" href="#table-of-contents-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Table of Contents: </h2>
<a href="#definition">- Problem Definition</a><br>
<a href="#description">- Data Description </a><br>
<a href="#objectives">- Objectives </a><br>
<a href="#analysis">- Exploratory Data Analysis</a><br>
<a href="#data-pre">- Data Preprocessing</a><br>
<a href="#modeling">- Modeling</a><br>
<a href="#deployment">- Deployment</a><br>
<br>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://camo.githubusercontent.com/9949fc233bb90d4c44117aa6b9f9f4ca3d34a8b081b8df801cdbced47d68de84/68747470733a2f2f67616d65736d69782e6e65742f77702d636f6e74656e742f75706c6f6164732f323032322f30332f464946412d32332e6a706567"><img src="https://camo.githubusercontent.com/9949fc233bb90d4c44117aa6b9f9f4ca3d34a8b081b8df801cdbced47d68de84/68747470733a2f2f67616d65736d69782e6e65742f77702d636f6e74656e742f75706c6f6164732f323032322f30332f464946412d32332e6a706567" data-canonical-src="https://gamesmix.net/wp-content/uploads/2022/03/FIFA-23.jpeg" style="max-width: 100%;"></a></p>
<h2 id="user-content-definition" dir="auto"><a id="user-content-problem-definition-" class="anchor" aria-hidden="true" href="#problem-definition-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Problem Definition: </h2>
<p dir="auto">Innovation Campus Club&nbsp;is a new professional football club, that wants to Compete Against the Top Clubs.</p>
<ul dir="auto">
<li>The club board knows how Data Analysis and Machine Learning can help them learn more about the Skills that need to be in their Players,
the top Clubs that they need to compete in, and the Best Position of the Players Based on their skills and know the similarity of the Players
in their Team so they can create a strong team and ensure that each player will play efficiently in his Position.</li>
</ul>
<h2 id="user-content-description" dir="auto"><a id="user-content-data-description-" class="anchor" aria-hidden="true" href="#data-description-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Data Description: </h2>
<p dir="auto">The Data Contains:</p>
<ul dir="auto">
<li>Every player available in FIFA 23</li>
<li>90 attributes</li>
<li>Player best position, with the role in the club and in the national team</li>
<li>Player attributes with statistics as Attacking, Skills, Defense, Mentality, GK Skills, etc.</li>
<li>Player personal data like Nationality, Club, DateOfBirth, Wage, Salary, etc.</li>
</ul>
<h4 dir="auto"><a id="user-content-you-can-find-the-data-on-kaggle---here" class="anchor" aria-hidden="true" href="#you-can-find-the-data-on-kaggle---here"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>You Can Find the Data on Kaggle --&gt;<a href="https://www.kaggle.com/datasets/cashncarry/fifa-23-complete-player-dataset" rel="nofollow">Here</a></h4>
<h2 id="user-content-objectives" dir="auto"><a id="user-content-objectives-" class="anchor" aria-hidden="true" href="#objectives-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Objectives: </h2>
<p dir="auto">1- Helping the club board know the best players in the different Clubs.</p>
<p dir="auto">2- Helping them Understand their competitor’s Clubs.</p>
<p dir="auto">3- Knowing the skills that need to be in their players.</p>
<p dir="auto">4- Helping them put the players in their suitable Position.</p>
<p dir="auto">5- Grouping the Club Players in Groups.</p>
<h2 id="user-content-analysis" dir="auto"><a id="user-content-exploratory-data-analysis-" class="anchor" aria-hidden="true" href="#exploratory-data-analysis-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Exploratory Data Analysis: </h2>
<h3 dir="auto"><a id="user-content-questions-to-be-answered--" class="anchor" aria-hidden="true" href="#questions-to-be-answered--"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Questions to be Answered : </h3>
<p dir="auto">1- Does the Age of the Player Affect on his Ball Control Performance?<br></p>
<p dir="auto">2- How Height affects different factors like stamina, dribbling, pace, passing and Heading Accuracy?<br></p>
<p dir="auto">3- Show if there is a relation between Wage and Overall of the Players.<br></p>
<p dir="auto">4- Show the Fastest Players.<br></p>
<p dir="auto">5- Determine if there is a relation between the Position of the Player, his Wage, and his Value.<br></p>
<p dir="auto">6- See the Nationality of the Players that got the highest Wages.<br></p>
<p dir="auto">7- Show the effect of the Age on the Potential of the Players.<br></p>
<p dir="auto">8- View the Top 50 Players and their Clubs <br></p>
<h4 dir="auto"><a id="user-content-you-can-find-the-eda-notebook---here" class="anchor" aria-hidden="true" href="#you-can-find-the-eda-notebook---here"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>You Can Find the EDA notebook --&gt;<a href="https://github.com/miraehab/FIFA-23-ML-Project/blob/main/EDA.ipynb">Here</a></h4>
<h2 id="user-content-data-pre" dir="auto"><a id="user-content-data-preprocessing-" class="anchor" aria-hidden="true" href="#data-preprocessing-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Data Preprocessing: </h2>
<h3 dir="auto"><a id="user-content--steps--" class="anchor" aria-hidden="true" href="#-steps--"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a> Steps : </h3>
1- Handle the missing values<br>
2- Handle The Categorical Columns<br>
3- Handle the Imbalanced Data<br>
4- Feature Scaling<br>
<h4 dir="auto"><a id="user-content-you-can-find-the-data-preprecessing-notebook---here" class="anchor" aria-hidden="true" href="#you-can-find-the-data-preprecessing-notebook---here"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>You Can Find the Data Preprecessing notebook --&gt;<a href="https://github.com/miraehab/FIFA-23-ML-Project/blob/main/dataPreprocessing.ipynb">Here</a></h4>
<h2 id="user-content-modeling" dir="auto"><a id="user-content-modeling-" class="anchor" aria-hidden="true" href="#modeling-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Modeling: </h2>
<p dir="auto">A. Predict the Position of the Player Using 8 Classification Algorithms --&gt; <a href="https://github.com/miraehab/FIFA-23-ML-Project/blob/main/PositionClassification.ipynb">NoteBook</a> <br></p>
<p dir="auto">B. Group the Players in Clusters Based on their Similarities Using 4 Clustering Algorithms --&gt; <a href="https://github.com/miraehab/FIFA-23-ML-Project/blob/main/Clustering.ipynb">NoteBook</a> <br></p>
<h2 id="user-content-deployment" dir="auto"><a id="user-content-deployment-" class="anchor" aria-hidden="true" href="#deployment-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Deployment: </h2>
<p dir="auto">Deployed the Classification Model using Flask and make HTML pages to test the predictions.</p>
<h4 dir="auto"><a id="user-content-you-can-find-the-deployment-files---here" class="anchor" aria-hidden="true" href="#you-can-find-the-deployment-files---here"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>You Can Find the Deployment Files --&gt;<a href="https://github.com/miraehab/FIFA-23-ML-Project/tree/main/Deployment">Here</a></h4>
<h3 dir="auto"><a id="user-content-screenshots" class="anchor" aria-hidden="true" href="#screenshots"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>ScreenShots:</h3>
<p dir="auto"><a target="_blank" rel="noopener noreferrer nofollow" href="https://user-images.githubusercontent.com/74511706/200836542-1d07a5a9-bd09-470a-ac95-45cca91f7582.png"><img src="https://user-images.githubusercontent.com/74511706/200836542-1d07a5a9-bd09-470a-ac95-45cca91f7582.png" alt="image" style="max-width: 100%;"></a>
<a target="_blank" rel="noopener noreferrer nofollow" href="https://user-images.githubusercontent.com/74511706/200836567-f267235a-a45b-4d85-a458-dc57351123bc.png"><img src="https://user-images.githubusercontent.com/74511706/200836567-f267235a-a45b-4d85-a458-dc57351123bc.png" alt="image" style="max-width: 100%;"></a>
<a target="_blank" rel="noopener noreferrer nofollow" href="https://user-images.githubusercontent.com/74511706/200836590-68f1b31d-001e-418c-bde3-43055dadb536.png"><img src="https://user-images.githubusercontent.com/74511706/200836590-68f1b31d-001e-418c-bde3-43055dadb536.png" alt="image" style="max-width: 100%;"></a>
<a target="_blank" rel="noopener noreferrer nofollow" href="https://user-images.githubusercontent.com/74511706/200836628-b3d1db8e-024a-436b-9c51-161bc3569728.png"><img src="https://user-images.githubusercontent.com/74511706/200836628-b3d1db8e-024a-436b-9c51-161bc3569728.png" alt="image" style="max-width: 100%;"></a></p>
