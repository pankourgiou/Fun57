from pykeen.pipeline import pipeline
from pykeen.models.uncertainty import predict_hrt_uncertain

# train model
# note: as this is an example, the model is only trained for a few epochs,
#       but not until convergence. In practice, you would usually first verify that
#       the model is sufficiently good in prediction, before looking at uncertainty scores
result = pipeline(dataset="nations", model="ERMLPE", loss="bcewithlogits")

# predict triple scores with uncertainty
prediction_with_uncertainty = predict_hrt_uncertain(
    model=result.model,
    hrt_batch=result.training.mapped_triples[0:8],
)

# use a larger number of samples, to increase quality of uncertainty estimate
prediction_with_uncertainty = predict_hrt_uncertain(
    model=result.model,
    hrt_batch=result.training.mapped_triples[0:8],
    num_samples=100,
)

# get most and least uncertain prediction on training set
prediction_with_uncertainty = predict_hrt_uncertain(
    model=result.model,
    hrt_batch=result.training.mapped_triples,
    num_samples=100,
)
df = result.training.tensor_to_df(
    result.training.mapped_triples,
    logits=prediction_with_uncertainty.score[:, 0],
    probability=prediction_with_uncertainty.score[:, 0].sigmoid(),
    uncertainty=prediction_with_uncertainty.uncertainty[:, 0],
)
print(df.nlargest(5, columns="uncertainty"))
print(df.nsmallest(5, columns="uncertainty"))
