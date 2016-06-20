import vessel_scoring.models
import vessel_scoring.utils
import gpsdio
import numpy
import sys

models = vessel_scoring.models.load_models()

model = sys.argv[1]
input = sys.argv[2]
output = sys.argv[3]

if input.endswith(".msg"):
    with gpsdio.open(output, "w") as fout:
        with gpsdio.open(input) as fin:
            fout.write(models[model].predict_messages(fin))
else:
    data = numpy.load(input)['x']
    datalen = len(data)
    data = vessel_scoring.utils.numpy_to_messages(data)
    data = models[model].predict_messages(data)
    data = vessel_scoring.utils.messages_to_numpy(data, datalen)
    numpy.savez(output, x=data)
