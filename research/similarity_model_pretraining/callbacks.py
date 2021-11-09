# coding=utf-8
# Copyright 2021 TF-Transformers Authors.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf

all_questions = [
    'Could someone still see my profile picture if they blocked me on WhatsApp?',
    'How do I get rid of sores in mouth?',
    "Who's the best US president ever?",
    'How do I treat extremely high blood pressure?',
    'What is Facebook video upload limit?',
    'Is the AMD FX-4100 3.60 GHz Quad good for gaming?',
    'How is Lipton Green Tea related to weight loss?',
    "Why don't amateurs launch a rocket into space, orbit, or the moon?",
    'What is the average salary of a MBA in India?',
    'What can I do if my wife and my mother hate each other?',
    'Can TQWL tickets confirmd easily?',
    'Which tablet have a usb slot in india?',
    'How can I reduce my chubby tummy?',
    'How do I go about creating a new language?',
    'Do employees at USG have a good work-life balance? Does this differ across positions and departments?',
    'When will bleach final arc episodes will release?',
    'Will the decision to demonetize 500 and 1000 rupee notes help to curb black money?',
    'What does the typical financial projection for a mobile or web startup look like?',
    'What are the uses of money market instruments?',
    'How can you lose weight fast in a healthy way?',
    'What are some tips on making it through the job interview process at Cardinal Financial?',
    'How is the relative ratio of brain waves (alpha/beta/gamma/delta/theta) different between fish and humans?',
    'How are FBI special agents trained?',
    'What are the best ways to come out of depression?',
    'Which are the best hair salons for hair spa in Pimple Saudagar, Pune?',
    'What is life like in a supermax prison?',
    'Who can be the next chairman of TATA?',
    'Would you marry a girl who is not virgin?',
    'Can I still smoke weed with getting drug tested every week with a 20 ng/ml test?',
    'What was your most embarrassing experience?',
    'Why do dogs bark at rag-pickers?',
    'How do I recover/reset my AOL email password?',
    'Does the "suggested friends" box on Facebook prefer showing the people who have recently looked at your profile?',
    'What universities does Target recruit new grads from? What majors are they looking for?',
    'What is the purpose of human life with more than 7.2 billion souls on this planet?',
    "What is Die Antwoord's best song? What are the lyrics and what do they mean?",
    'How is the upcoming market demand of node.js?',
    'Why is time defined as a dimension?',
    'Hat should I do to score more than 99 percentile in CAT 2016 , if I start preparing now?',
    'Does Hillary Clinton think she can be fair in the Palestine-Israel conflict when she unequivocally supports Israel?',
    'What are some dirty secrets of Bollywood?',
    "In the Lord Of The Rings: was Sauron in the form of an 'eye' as shown in the movies or was \
        he in the form of a spirit/ghost?",
    'I did not maintain the minimum balance in HDFC and now my balance is negative (i.e. -5000). \
        I cannot close it until I pay it. Anyways, I have an account in other banks. Will this affect my \
            CIBIL score? Will this affect me in the future or shall I just avoid it?',
    'How can I increase my intelligence as much as possible?',
    'What book did President Obama use when he taught Con Law?',
    "Can you explain the darwin's correction in fine structure of hydrogen atom?",
    'What is the next big thing in medicine?',
    'Is a gross salary of DKK 40,000 a month good for a expat Software Engineer in Copenhagen?',
    'How does Quora decide what answers to collapse?',
    'Why do Indians have arranged marriages?',
    'What are the biggest differences between the Jeep Wrangler trims?',
    'Is Curiosity (Mars rover) planning to record any video footage? Can it?',
    "What is the lowest seed an NBA regular season MVP's team has ever been?",
    'What are the effects of long term sleep deprivation?',
    'What is the most profitable agricultural crop in maharashtra?',
    'How do all electronics work?',
    'If I block someone in my Instagram and I decide to send a direct \
        message, can she/he read that DM even if I block her/him?',
    'Why is it better to be a fan of Microsoft rather than a fan of Google?',
    'How can I make money?',
    'What thought scares you the most?',
    'What is sensex? What is nifty?',
    'Should I just concentrate on AI and forget Data Science?',
    'Should I start packing, now that Trump won the presidency?',
    "What are some mind blowing Safety wallets that most people don't know about?",
    "What would be Hillary Clinton's foreign policy towards India if elected as the President of United States?",
    'What are class 12 CBSE board exam tips and suggestions?',
    'What are the pros and cons of being a driver for Uber or Lyft in Seattle?',
    "What would be Hillary Clinton's foreign policy towards India if elected as the President of United States?",
    'If the purge were to start tonight, what would your plan be to survive?',
    'What is the best passive investment strategy?',
    "What'd be the top (maximum) wavelength of an electromagnetic radiation to see a message written in a paper?",
    'What is the difference between satisfaction and pleasure?',
    'What are some of the best business books?',
    'What is the scope of doing MBA in information management?',
    'What is your dark side?',
    'How do I lose stubborn belly fat?',
    'If deposits of iron ore came from supernovae, why are they concentrated in geographic regions?',
    'What is a way to convince your parents to get another dog?',
    'Under what circumstances will ppc (Production possibility curve) be convex to origin?',
    "My 4K TV doesn't support HEVC. What can I do to stream 4K videos?",
    "What is the difference between Dhoni's captaincy and Kohli's captaincy?",
    'How is black money curbed with the ban of 1000 rupee notes and introducing new 500 and 2000 rupee notes?',
    'Should Quora ban Peter Johnson?',
    "What's a polite way to let people know that I don't want presents of specific type?",
    'What are some classic Indian recipes for chicken? B',
    'Can old diapers be reused?',
    'How do I solve SPOJ MOEBIUS?',
    'What is physical quantities and also its types?',
    'Which department in HPCL is better to work in?',
    'What was the significance of the battle of Somme, and how did this battle compare \
        and contrast to the Battle of Yalu River?',
    'During British India how did the British officers living faraway from home, wife and children handle \
        themselves?',
    'What are your views on ban of 500 and 1000 rupee notes in India?',
    'Why is it difficult to get a job in India?',
    'Will Donald Trump build a wall?',
    'What would be the first thing you do as a zombie apocalypse survivor?',
    'How can I control my sugar?',
    'How do I love my job?',
    'What is your favorite quote from Dota 2?',
    'What is your spirit or soul consisted of?',
    'If a black hole grows larger as it absorbs more of the universe, can it absorb the \
        entire universe given enough time?',
    "What does it feel like to think you're physically attractive, yet have poor success with dating?",
    'What fuel consumption should I expect for a 45 ton gross weight tipper truck (4x2)?',
    'Is RBI really launching new 2000 rupee notes?',
    'Which team should I join on Pokémon GO?',
    'What are the best free resources to learn Java?',
    "I filed for divorce but my spouse went into hiding so I can't serve the divorce papers. \
        What is the best route of action in this case?",
    'What is the comparison between AVG Antivirus and AVG Internet Security?',
    'Who is the richest gambler of all time?',
    'Did the Rapture happen?',
    'What is the Sahara, and how do the average temperatures there compare to the ones in the Dasht-e Margo?',
    'What is the difference between a "wrong" and a "tort"?',
    'How is Reliance Jio providing free unlimited 4G data when other companies charge high?',
    'Will Hillary Clinton cause WWIII by going to war with Syria?',
    'What if India bans import of all Chinese products?',
    'What is the direction of qibla in America?',
    'What are some tips for starting an Etsy shop?',
    'What is it like to be an undergraduate international student at university of melbourne?',
    'What is it like to be a bus driver?',
    "Citizenship: What's the best country in the world for a person to be born, and live in?",
    'How can I grow taller?',
    'What is the superstition behind a twitching left eye?',
    'How do I earn a higher grade in Calculus 2?',
    'What are some funny rules for dating?',
    'Is there a payroll service as delightful as Servicejoy.com for invoicing?',
    'What are some theories of time travel?',
    'What are 25 random questions to ask someone you just met?',
    'What are the best "safety" universities for undergraduate CS in the United States?',
    'What do modern Japanese think of "The Mikado” by Gilbert and Sullivan?',
]

# Callbacks


class SimilarityCallback:
    """Simple MLM Callback to check progress of the training"""

    def __init__(self, top_k=5):
        """Init"""
        self.all_questions = all_questions
        self.top_k = top_k

    def __call__(self, trainer_params):
        """Main Call"""
        model = trainer_params['model']
        validation_dataset = trainer_params['validation_dataset']

        original_embeddings = []
        corrupted_embeddings = []
        for batch_inputs, _batch_labels in validation_dataset:
            model_outputs = model(batch_inputs)
            original_embeddings.append(model_outputs['original_sentence_embedding_normalized'].numpy())
            corrupted_embeddings.append(model_outputs['corrupted_sentence_embedding_normalized'].numpy())

        original_embeddings = np.vstack(original_embeddings)
        corrupted_embeddings = np.vstack(corrupted_embeddings)

        logits_softmax = tf.matmul(original_embeddings, corrupted_embeddings, transpose_b=True)
        top_prob, top_k = tf.nn.top_k(logits_softmax, k=self.top_k)

        similar_texts = []
        similar_probs = []
        for i in range(top_prob.shape[0]):

            probs = top_prob[i]
            indexes = top_k[i]
            similar_texts.append([self.all_questions[index] for index in indexes])
            similar_probs.append(probs.numpy().tolist())

        df = pd.DataFrame(similar_texts)
        wandb = trainer_params['wandb']
        global_step = trainer_params['global_step']
        # Log to wandb as a table
        if wandb:
            wandb.log({"similarity_table_step_{}".format(global_step): wandb.Table(dataframe=df)}, step=global_step)
        else:
            print(df)
