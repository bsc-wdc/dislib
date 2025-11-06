from pycompss.api.parameter import IN, COLLECTION_IN
from pycompss.api.constraint import constraint
from pycompss.api.task import task
import math
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch


@constraint(processors=[
        {'processorType': 'CPU', 'computingUnits': '1'},
        {'processorType': 'GPU', 'computingUnits': '1'}])
@task(x_test=COLLECTION_IN, model=IN,
      parameters_for_model=COLLECTION_IN, returns=2)
def evaluate_model_parameters(x_val, y_val, model,
                              parameters_for_model,
                              loss_function,
                              score_function=accuracy_score,
                              y_val_loss_function=None):
    model = assign_parameters(model, parameters_for_model)
    validation_loss = None
    validation_acc = None
    if x_val is not None:
        torch_model = model.eval().to("cuda:0")
        indexes = 128
        num_batches = math.ceil(x_val[0][0].shape[0]/indexes)
        outputs = []
        for x_out_tens in x_val:
            for x_in_tens in x_out_tens:
                x_in_tens = x_in_tens.to("cuda:0")
                for idx in range(num_batches):
                    with torch.no_grad():
                        output = torch_model(x_in_tens
                                             [idx * indexes:
                                              (idx + 1) * indexes].
                                             float())
                        output_cpu = output.cpu()
                        outputs.append(output_cpu)
                        del output
                    torch.cuda.empty_cache()
                x_in_tens = x_in_tens.to("cpu")
                del x_in_tens
                torch.cuda.empty_cache()
        outputs = torch.cat(outputs)
        loss = loss_function()
        if y_val_loss_function is not None:
            validation_loss = loss(outputs, y_val_loss_function).item()
        else:
            validation_loss = loss(outputs, y_val).item()
        outputs = process_outputs(outputs)
        outputs = outputs.detach().cpu().numpy()
        validation_acc = score_function(y_val, outputs)
        del torch_model
    return validation_loss, validation_acc


def pt_aggregate_parameters(workers_parameters):
    NUM_WORKERS = len(workers_parameters)

    final_weights = []
    for i in range(NUM_WORKERS):
        workers_weights = []
        for param in workers_parameters[i].parameters():
            workers_weights.append(param)
        final_weights.append(workers_weights)
    final_added_parameters = final_weights[0]
    for i in range(len(final_weights[0])):
        for j in range(1, len(final_weights)):
            final_added_parameters[i] = final_added_parameters[i] + \
                                        final_weights[j][i]

    for i in range(len(final_weights[0])):
        final_added_parameters[i] = final_added_parameters[i]/NUM_WORKERS
    j = 0
    if hasattr(workers_parameters[0], 'neural_network_layers'):
        len_nn = len(workers_parameters[0].neural_network_layers)
        for i in range(len_nn):
            if hasattr(workers_parameters[0].neural_network_layers[i],
                       'weight'):
                workers_parameters[0].neural_network_layers[i].weight = \
                    nn.Parameter(final_added_parameters[j])
                j += 1
                workers_parameters[0].neural_network_layers[i].bias = \
                    nn.Parameter(final_added_parameters[j])
                j += 1
            if hasattr(workers_parameters[0].neural_network_layers[i],
                       'shortcut'):
                len_shortcut = len(workers_parameters[0].
                                   neural_network_layers[i].shortcut)
                for k in range(len_shortcut):
                    if hasattr(workers_parameters[0].
                               neural_network_layers[i].shortcut[k],
                               'weight'):
                        workers_parameters[0].\
                                neural_network_layers[i].\
                                shortcut[k].weight = \
                                nn.Parameter(final_added_parameters[j])
                        j += 1
                        workers_parameters[0].\
                            neural_network_layers[i].\
                            shortcut[k].bias = \
                            nn.Parameter(final_added_parameters[j])
                        j += 1
                    if hasattr(workers_parameters[0].
                               neural_network_layers[i].shortcut[k],
                               'alpha'):
                        workers_parameters[0].\
                            neural_network_layers[i].\
                            shortcut[k].alpha = \
                            nn.Parameter(final_added_parameters[j])
                        j += 1
            if hasattr(workers_parameters[0].
                       neural_network_layers[i],
                       'layers'):
                len_layers = len(workers_parameters[0].
                                 neural_network_layers[i].layers)
                for k in range(len_layers):
                    if hasattr(workers_parameters[0].
                               neural_network_layers[i].layers[k],
                               'weight'):
                        workers_parameters[0].\
                                neural_network_layers[i].layers[k].weight = \
                                nn.Parameter(final_added_parameters[j])
                        j += 1
                        workers_parameters[0].\
                            neural_network_layers[i].layers[k].bias = \
                            nn.Parameter(final_added_parameters[j])
                        j += 1
                    if hasattr(workers_parameters[0].
                               neural_network_layers[i].layers[k],
                               'alpha'):
                        workers_parameters[0].\
                                neural_network_layers[i].layers[k].\
                                alpha = \
                                nn.Parameter(final_added_parameters[j])
                        j += 1
    if hasattr(workers_parameters[0], 'dense_neural_network_layers'):
        len_nn = len(workers_parameters[0].dense_neural_network_layers)
        aux_j = 0
        for i in range(len_nn):
            if hasattr(workers_parameters[0].dense_neural_network_layers[i],
                       'weight'):
                workers_parameters[0].dense_neural_network_layers[i].weight = \
                    nn.Parameter(final_added_parameters[aux_j + j])
                aux_j += 1
                workers_parameters[0].dense_neural_network_layers[i].bias = \
                    nn.Parameter(final_added_parameters[aux_j + j])
                aux_j += 1
    return workers_parameters[0]


def assign_parameters(model, trained_weights):
    j = 0
    if hasattr(model, 'neural_network_layers'):
        len_nn = len(model.neural_network_layers)
        for i in range(len_nn):
            if hasattr(model.neural_network_layers[i], 'weight'):
                model.neural_network_layers[i].weight = \
                        nn.Parameter(trained_weights.
                                     neural_network_layers[i].weight)
                j += 1
                model.neural_network_layers[i].bias = \
                    nn.Parameter(trained_weights.
                                 neural_network_layers[i].bias)
                j += 1
            if hasattr(model.neural_network_layers[i], 'shortcut'):
                len_shortcut = len(model.neural_network_layers[i].shortcut)
                for k in range(len_shortcut):
                    if hasattr(model.neural_network_layers[i].shortcut[k],
                               'weight'):
                        model.neural_network_layers[i].shortcut[k].weight = \
                            nn.Parameter(trained_weights.
                                         neural_network_layers[i].
                                         shortcut[k].weight)
                        j += 1
                        model.neural_network_layers[i].shortcut[k].bias = \
                            nn.Parameter(trained_weights.
                                         neural_network_layers[i].
                                         shortcut[k].bias)
                        j += 1
                    if hasattr(model.neural_network_layers[i].shortcut[k],
                               'alpha'):
                        model.neural_network_layers[i].shortcut[k].alpha = \
                            nn.Parameter(trained_weights.
                                         neural_network_layers[i].
                                         shortcut[k].alpha)
                        j += 1
            if hasattr(model.neural_network_layers[i],
                       'layers'):
                len_layers = len(model.neural_network_layers[i].layers)
                for k in range(len_layers):
                    if hasattr(model.neural_network_layers[i].layers[k],
                               'weight'):
                        model.neural_network_layers[i].layers[k].weight = \
                            nn.Parameter(trained_weights.
                                         neural_network_layers[i].
                                         layers[k].weight)
                        j += 1
                        model.neural_network_layers[i].layers[k].bias = \
                            nn.Parameter(trained_weights.
                                         neural_network_layers[i].
                                         layers[k].bias)
                        j += 1
                    if hasattr(model.neural_network_layers[i].layers[k],
                               'alpha'):
                        model.neural_network_layers[i].layers[k].alpha = \
                            nn.Parameter(trained_weights.
                                         neural_network_layers[i].
                                         layers[k].alpha)
                        j += 1
    if hasattr(model, 'dense_neural_network_layers'):
        len_nn = len(model.dense_neural_network_layers)
        aux_j = 0
        for i in range(len_nn):
            if hasattr(model.dense_neural_network_layers[i], 'weight'):
                model.dense_neural_network_layers[i].weight = \
                        nn.Parameter(trained_weights.
                                     dense_neural_network_layers[i].
                                     weight)
                aux_j += 1
                model.dense_neural_network_layers[i].bias = \
                    nn.Parameter(trained_weights.
                                 dense_neural_network_layers[i].
                                 bias)
                aux_j += 1
    return model


def process_outputs(output_nn):
    _, indices = torch.max(output_nn, dim=1)
    binary_output = torch.zeros_like(output_nn)
    binary_output[torch.arange(output_nn.size(0)), indices] = 1
    return binary_output


def compute_validation_losses(torch_model, x_val, y_val_to_loss,
                              y_val, loss_function,
                              validation_score=accuracy_score):
    validation_loss = []
    validation_acc = []
    torch_model = torch_model.eval().to("cuda:0")
    indexes = 128
    num_batches = math.ceil(x_val[0][0].tensor_shape[0]/indexes)
    outputs = []
    for x_out_tens in x_val:
        for x_in_tens in x_out_tens:
            x_in_tens = x_in_tens.to("cuda:0")
            for idx in range(num_batches):
                with torch.no_grad():
                    output = torch_model(x_in_tens[idx * indexes:
                                         (idx + 1) * indexes].
                                         float())
                    output_cpu = output.cpu()
                    outputs.append(output_cpu)
                    del output
                torch.cuda.empty_cache()
            x_in_tens = x_in_tens.to("cpu")
            del x_in_tens
            torch.cuda.empty_cache()
    outputs = torch.cat(outputs)
    loss = loss_function()
    validation_loss.append(loss(outputs, y_val_to_loss).item())
    if outputs.shape[-1] > 1:
        outputs = process_outputs(outputs)
    outputs = outputs.detach().cpu().numpy()
    validation_acc.append(accuracy_score(y_val, outputs))
    return validation_loss, validation_acc
