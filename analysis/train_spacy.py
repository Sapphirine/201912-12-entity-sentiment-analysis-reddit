# training the spacy model

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# new entity label
LABEL = "COMPANY"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    (
        "Anyway facebook is dead. All the interesting people left the platform",
        {"entities": [(7, 15, LABEL)]},
    ),
    ("When Facebook was founded, I bet a lot of sites were still using plain text.", {"entities":[(5, 13, LABEL)]}),
    ("Facebook is both a platform and a news company.", {"entities":[(0, 8, LABEL)]}),
    ("I despise Facebook and corporations, but that doesn’t negate any facts at all.", {"entities":[(10, 18, LABEL)]}),
    ("The Facebook group suggests that regardless of who made those rules, they enforced them gleefully.", {"entities": []}),

    ("Apps that use Facebook to track their users require those users to install their apps.", {"entities": [(14, 22, LABEL)]}),
    (
        "Yet it won't matter. Facebook has reached that critical mass where people treat it like a utility they don't want to live without. I've known many people who talk about all the shit Facebook has done while they are actively checking Facebook. Sure, some people have abandoned the platform, but not many and not nearly enough for it to make a difference.",
        {"entities": [(21, 29, LABEL), (182, 190, LABEL), (233, 241, LABEL)]},
    ),
    ("Ah dude. Never give facebook your address.", {"entities": [(20, 28, LABEL)]}),
    ("I came across an article about green tea on facebook where they showed it's not healthy as you might think", {"entities": []}),
    ("Facebook tracking is not a new concept. It's a digital version of a decades long practice.", {"entities": [(0, 8, LABEL)]}),
    ("Trump's live feed on facebook was the funniest thing ever.", {"entities": []}),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="animal", output_dir="/Users/candide/Acads/BIG DATA/final_project/reddit-sentiment/spacy_fb", n_iter=100):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "Facebook tracking is not a new concept. It's a digital version of a decades long practice."
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
