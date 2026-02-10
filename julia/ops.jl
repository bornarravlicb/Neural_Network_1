#Activations
module Activations
export linear, relu, sigmoid, tanh

linear(x) = x

relu(x) = max.(0, x)

sigmoid(x) = 1 ./ (1 .+ exp.(-x))

tanh(x) = (exp.(x) .- exp.(-x)) ./ (exp.(x) .+ exp.(-x))

end

#Layer processing
module Layers
export forward, backward

function forward(x, w, b)
    return w * x .+ b
end

function backward(dz, x, w)
    dW = dz * x'
    db = sum(dz, dims=2)
    dx = w' * dz
    return dx, dW, db
end

end

#Losses
module Losses
export mse, dmse
using Statistics

function mse(y_pred, y_true)
    return mean((y_pred .- y_true).^2)
end

function dmse(y_pred, y_true)
    return 2 .* (y_pred .- y_true) ./ length(y_true)
end

end

#Optimizers
module Optimizers
export update_params

function update_params(w, b, dw, db, lr)
    w .-= lr .* dw
    b .-= lr .* db
end

end