Requirements:
Dynet, numpy, nltk, argparse are to be installed

How to run:
1) Have included sample data, since the whole data is large and couldnt be included. For that need to download Enron corpus( https://www.cs.cmu.edu/~enron/ ) and run :
 python read_data.py  
 on the taylor-m folder only!
Now you have pre processed data for one inbox that I am considering.
 
2) to train and test : 

(dynet) $ python email_linguistic_signature_with_CV.py  --datapath "/work/pooja/dynet-base/dynet_tutorial/from_email_body_extracted/"  --train temp.model --test temp.model --num_epochs 2 --embedding_approach pretrained   

It automatically runs test on the specific fold and the model saved is re written each time.


