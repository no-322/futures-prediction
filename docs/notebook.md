# Project Notebook

## 2026-05-24 — Sunday

**Built** the scaffolding and foundation. I learnt proper usage of hooks and set up log files to record userprompts. Initially I assumed this would lead to deterministic action. But later realized the model responses are stochastic in nature. Although the log file doesn't guarantee replicability, it does give an audit trail that shows the thought process as well. 

Added [CLAUDE.md](./../CLAUDE.md) file to ensure that the methodology is specified and to avoid common traps during the machine learning process.

I caught claude hallucinating what the tick columns represents. Although it seemed reasonable enough, assumption on input data is dangerous.

Did some basic EDA with claude as well. An interesting point that came up is for some records there is a gap of more than 1 minute - 2 min, 3min, overnight, weekends, etc. *How do we handle this?* On one hand we could just consider the previous record. But even if market is not live there would have been changes to the underlying and capturing previous record might give us stale data. 

## 2026-05-25 — Monday

*Built* src/load.py, src/split.py. The shell files ensure that anytime a file in src is run, the relevant test files are autotriggered. Hence, first we create the test files. Then the src files and run it. 

*Mistake noticed:* I noticed that claude just prints a warning when datetime is not in monotonic order. But the downstream file(load.py) runs with it assuming it is not monotonic. So I changed it to ensure it is always sorted. I first tried it in `accept edits on` mode but it started to assert error when datetime was out of error. So reset to previous commit and did it in plan mode verifying the changes. 

## 2026-05-26 - Tuesday
In addition to this I also want a deeper understanding of everything I am coding, so I am currently studying Elements of Statistical Learning sections 7.1-7.4. 

*Built* features.py and labels.py to create features and labels. *Mistake Noticed:* Despite mentioning in [CLAUDE.md](../CLAUDE.md) claude failed to update function signatures in [MODULES.md](MODULES.md)

## 2026-05 - Wednesday
I realised I made a mistake. I had the order `load` -> `split` -> `feature extraction` -> `label`. But this pipeline would need a rerun as test dataset would need a rerun of `feature extraction`. So changed the pipeline order to `load` -> `feature extraction` -> `split`  -> `label`

## 2026-05-28 - Thursday

HTML was added to log.md. Seems to be a bug. I need to figure out why this is happening. I also learnt about joblib model persistence and how we wouldn't have to retrain everytime, this will be handy for more complex models like SVM. So I am going to add it for the other models as well. 

Turns out Claude injects XML into user message stream when a background task is run. When evaluation.py was written a background task of run all tests was trigerred. So this was logged in as well. It is interesting to note that while this is being logged, clarification questions asked post plan or during edit are not logged. 

## 2026-05-29 - Friday
Key result: the difference observed in accuracy vs prediction in SVM is because we balanced class only in SVM case. When we balance class accuracy goes down but precision increases. With class imbalance the model does not take risk and predicts a lot of down. 

## 2026-06-03 - Wednesday

Created pipeline.py that is an automated end to end model that runs based on selected model. We can use the CLI to give parameters on what model to run and change the other previously static parameters as well- random state is still fixed at 42. Created a yaml file that will act as the intermediary between the GUI and CLI. All the changes will be made in the config based on the user inputs. The parameters are then passed on to the pipeline and individual files.

Created the GUI. I wanted to have an info button that shows if a trained model already exists. But had issue because json file was needed that tracks models trained with GUI but not CLI. Reworking the pipeline so that we have a centralized metadata file to track all the available trained joblib files. 

## 2026-06-12 - Friday

I noticed that SVM not only takes time to train but also 3.5 hrs to predict 250k rows. On investigation I found that overtraining in SVM and random forest. SVM has nearly one support vector for every record. Similarly random forest is too deep. After dropping flat bars GBM is making a profit of 300+%. Logistic and GBM are the ones with least overfitting. But when we add a transaction cost of even 1bp this 300% doesn't persist. We get a drawdown of 100%. Adding regularization might help rf and svm. Current rf model goes 68-188 levels. Next step would be to introduce max_depth somewhere between 5-10. raise min_sample_leaf and use ccp_alpha for regularization.  

## 2026-06-22 Monday

I have added a walk-forward evaluation module. I know that there are better models present I want to better optimize my model. But if I were to train in-order to get better accuracy on test-set, I lose the "unseen" test set and I am essentially doing data-mining. So I have introduced the walk-forward module. I am using a rolling window for the training. The default is 3 mo train and 1 mo predict. Based on past experiment results that also match with [Interpretable Hypothess-Driven Trading](https://arxiv.org/html/2512.12924v1) I expect high volatility period accuracy to be better. 

I am focusing on the top 5 models in the model leaderboard. After some EDA an interesting result that has come up is that both logistic regressions placed 1st and 2nd in the leaderboard are individual winners in 14 windows each out of a total 36 windows. This could mean  based on regime analysis if I see a clear pattern on when tuned is beating logisitc regression I can write a pipleline that will use a mixture to get better accuracy.

## 2026-06-23 Tuesday

Today I am focusing on orderbook related features. I already implemented VWAP imbalance (VWAP/Close - 1). Now I added volume z-score and signed volume to signify direction based on close> open.

The results showed that it helped the non-linear model but linear-models performed worse. After normalizing volume and since signed volume can not be normalized I had to divide by rolling history of volume for every fold (Normalization removes established polarity) it affected the linear models to a lesser extent but it still ended up damaging them

## 2026-06-25 Thursday
Today I tried adding HMM to test how much regime awareness adds to my result. I have also seen research papers suggest high vol regime to be favourable. But before proceeding I wanted to correct the previous implementation. Instead of filtering, previous results used smoothing. This introduced look-ahead bias. To correct this, I started using filtering by which regime is assigned based solely on data up until t-1. I implemented two versions of HMM. Version 1 uses HMM as a gate and trades only during high volatility periods. This has a coverage of 53.7% and a no-flat accuracy of 54.82%. The other uses probability of High volatility as a feature. Both seem to perfrom better than the existing winners with the HMM gates performing the best. High volatility region has an annualized vol of 9.4% and low volatility region has an annualized vol of 7.2%. Although gate has higher accuracy it's backtest is considerably worse since it doesnt trade on all days. 

## 2026-07-01 Wednesday

None of the model improvements are statistically significant.

## 2026-07-13 Monday

Today I will be focusing on clean-up of repository. Although the modules.md has function signature, it does not have pseudocode tracking which led to me creating FEATURES.md. The entire point of this pipeline is to have an efficient machine that avoids double work. The past 2 week usage of my pipeline has led me to see a lot of flaws. I will start by modifying the modules.md creation. I will update CLAUDE.md for the same as well. 

Work done include the following:
1. Leaderboard walk-forward created as a seperate leaderboad based on avg window no-flat accuracy while also showing count of windows where it beat baseline
2. post-tool use hook to maintain modules.md and module-diagram.md
3. Dropped all flat metrics
4. Although we by default remove flats.. this is valid only on the label and not during feature creation. 
5. Modified the feature-engineering skill
6. Removed SVM completely. 

More detailed information can be found in log.md timestamp - 2026-07-13 15:08:33