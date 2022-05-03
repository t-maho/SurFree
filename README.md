You will find in this Git, the code to perform SurFree attack, based on the *Foolbox* library.

# SurFree: a fast surrogate-free black-box attack

Paper : https://arxiv.org/abs/2011.12807 

Machine learning classifiers are critically prone to evasion attacks. Adversarial examples are slightly modified inputs that are then misclassified, while remaining perceptively close to their originals. Last couple of years have witnessed a striking decrease in the amount of queries a black box attack submits to the target classifier, in order to forge adversarials. This particularly concerns the blackbox score-based setup, where the attacker has access to top predicted probabilites: the amount of queries went from to millions of to less than a thousand. 

This paper presents *SurFree*, a geometrical approach that achieves a similar drastic reduction in the amount of queries in the hardest setup: black box decision-based attacks (only the top-1 label is available). We first highlight that the most recent attacks in that setup, *HSJA*, *QEBA* and *GeoDA* all perform costly gradient surrogate estimations. *SurFree* proposes to bypass these, by instead focusing on careful trials along diverse directions, guided by precise indications of geometrical properties of the classifier decision boundaries. We motivate this geometric approach before performing a head-to-head comparison with previous attacks with the amount of queries as a first class citizen. We exhibit a faster distortion decay under low query amounts (few hundreds to a thousand), while remaining competitive at higher query budgets.


# Install

* Install requirements

```bash
pip install -r requirements
```

* Clone Foolbox Library (This github have been tested for *Foolbox* in version 3.2.1. )

```bash
git clone https://github.com/bethgelab/foolbox
```

* Move surfree.py in the folder *./foolbox/attacks/*:

* Update *./foolbox/attacks/__init__.py* with the following line:

```python
from .surfree import SurFree
```

* You can now install foolbox with the following command to run SurFree like any attacks in FoolBox:
```bash
python setup.py install
```


# Run

You can now run the main.py. It will perform the SurFree on the some Foolbox test image.
You can change the SurFree parameters by giving to him in parameter a *config.json* file.

# Citation

```
@misc{maho2020surfree,
      title={SurFree: a fast surrogate-free black-box attack}, 
      author={Thibault Maho and Teddy Furon and Erwan Le Merrer},
      year={2020},
      eprint={2011.12807},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```