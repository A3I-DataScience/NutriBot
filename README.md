# NutriBot

To contain the code of the Nutrition bot data challenge.

https://www.datais.es/dataton-sostenibilidad

## Project Scope

#### In Spanish

Optimización de dietas nutricionales con IA Generativa

El objetivo de este reto es desarrollar un modelo y una aplicación utilizando IA generativa para optimizar dietas nutricionales, garantizando una ingesta balanceada y adecuada para diferentes poblaciones. Los participantes deberán abordar el diseño de soluciones que no solo promuevan la salud, sino que también sean personalizables y accesibles, teniendo en cuenta las preferencias y necesidades nutricionales de los usuarios.

Los proyectos serán evaluados en función de la efectividad del modelo para crear dietas equilibradas, la innovación en el uso de IA generativa, la usabilidad de la aplicación desarrollada y la claridad de la presentación de resultados. Se recomienda utilizar los siguientes conjuntos de datos: Apparent Intake (based on household consumption and expenditure surveys) y Suite of Food Security Indicators 2.

#### In English

Optimization of nutritional diets with Generative AI

The objective of this challenge is to develop a model and an application using generative AI to optimize nutritional diets, guaranteeing a balanced and adequate intake for different populations. Participants will need to address the design of solutions that not only promote health, but are also customizable and accessible, taking into account the preferences and nutritional needs of users.

Projects will be evaluated based on the effectiveness of the model in creating balanced diets, innovation in the use of generative AI, the usability of the developed application and the clarity of the presentation of results. It is recommended to use the following data sets: Apparent Intake (based on household consumption and expenditure surveys) and Suite of Food Security Indicators 2.

## Evaluation

You will have a maximum of 20 minutes to present your project, followed by a round of questions from the jury.

### Evaluation Criteria

These are the points on which the evaluation of your project will be based:

#### Quality of the Technical Solution (up to 40 points)

Innovation in the solution and approach to address the sustainability problem.
Feasibility and scalability of the solution in the real world.
Coverage of the complete data cycle: collection, analysis, integration and generation of insights.
Implementation of Machine Learning & AI.

#### Business Impact (up to 40 points)

Relevance and applicability of the solution for the sustainability sector and its recipients.
Ability to solve the problem in practice and potential to improve processes.
User Experience (UX) focused on ease of use, clarity in presentation of results and applicability.
Process improvement through technology and data for sustainable optimization.

#### Presentation and Defense (up to 20 points)

Quality and originality of the presentation and clarity of the message.
Clarity and effectiveness in the visualization of results.
Power of the message and ability to convince the jury.

## Data sets:

* https://www.fao.org/faostat/en/#data/HCES
* https://www.fao.org/faostat/en/#data/FS
* https://www.fao.org/faostat/en/#data

## Nutrition Guidelines References:

* https://iris.who.int/bitstream/handle/10665/326261/9789241515856-eng.pdf
* https://applications.emro.who.int/docs/EMROPUB_2019_en_23536.pdf?ua=1
* https://files.magicapp.org/guideline/a3fe934f-6516-460d-902f-e1c7bbcec034/published_guideline_7330-1_1.pdf

## Tezchical references

* https://python.langchain.com/docs/introduction/
* https://huggingface.co/learn/cookbook/en/rag_evaluation

## How to run 

1. clone the repo,
2. Create an Open AI Key the key [here](https://platform.openai.com/api-keys), and run in terminal
```
export OPENAI_API_KEY="your_api_key_here"
```
**Warning** there multiple types of key that you can create on open AI platform 

3. run 

```
pip install -r requirements.txt
```

4. run 
```
flask --app server.py --debug run
```



