# LinkedIn post

Attach `docs/media/replay.mp4` (or the GIF) as the video. Put the GitHub link in the first comment
if you prefer to keep the post body link-free.

## Français

Peut-on détecter un vol à l'étalage sans jamais stocker un visage ?

En 2024, la CNIL a jugé non conforme le principal logiciel français de détection de vol par caméra, notamment parce qu'il analyse en continu l'image des clients. J'ai voulu voir ce qu'on obtient en ne gardant que des squelettes.

Vigie transforme la vidéo en points du corps (YOLO11-pose) et suit chaque personne. Trois modèles lisent ensuite ces squelettes :
- un transformer qui a appris les gestes normaux sur des heures d'achats ordinaires ;
- 14 indices lisibles, comme « main du rayon vers le corps » ou « main gardée à la poche 1,8 s » ;
- un classifieur affiné sur 600 vols mis en scène.

Sur 47 vrais vols filmés dans un magasin (jeu de test PoseLift), il atteint 71,8 à 74,4 d'AUC, contre 69,2 pour la meilleure méthode publiée. Une nuance importante : ces méthodes n'ont jamais vu de vol, alors que Vigie a appris sur des vols mis en scène.

Le point faible est en tête du README : à une fausse alerte par heure et par caméra, Vigie ne repère que 12 % des vols. C'est aujourd'hui un outil pour signaler un passage à revoir, pas une alarme autonome.

Tout tourne sur le processeur d'un PC portable, sans carte graphique. La vidéo montre de vrais vols rejoués en squelettes, avec la courbe de risque et la période du vol en rouge.

Code, résultats et rapport technique : https://github.com/Polluxgnr/vigie

Pour ce genre de système, qu'est-ce qui compte le plus pour vous : la précision, ou ne rien stocker ?

#ComputerVision #PoseEstimation #MachineLearning #PrivacyByDesign #Retail

## English

Can you detect shoplifting without ever storing a face?

In 2024 the French privacy regulator (CNIL) found the leading French camera-based theft detector non-compliant, largely because it continuously analyses customers' images. I wanted to see how far you get when you keep only skeletons.

Vigie turns video into body keypoints (YOLO11-pose) and tracks each person. Three models then read the skeletons:
- a transformer that learned ordinary movement from hours of normal shopping;
- 14 readable cues, such as "hand went from shelf to body" or "hand kept at pocket 1.8 s";
- a classifier fine-tuned on 600 staged thefts.

On 47 real thefts filmed in a store (the PoseLift test set), it reaches 71.8 to 74.4 AUC, against 69.2 for the best published method. One important caveat: those methods never see a theft, while Vigie learned from staged ones.

The weak spot is at the top of the README: at one false alert per camera-hour, Vigie catches only 12% of thefts. Today it is a "flag this clip for review" tool, not an autonomous alarm.

Everything runs on a laptop CPU, no GPU. The video shows real thefts replayed as skeletons, with the risk curve and the theft period in red.

Code, results and technical report: https://github.com/Polluxgnr/vigie

For this kind of system, what matters more to you: accuracy, or storing nothing?

#ComputerVision #PoseEstimation #MachineLearning #PrivacyByDesign #Retail
