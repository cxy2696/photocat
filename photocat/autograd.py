"""A compact reverse-mode automatic differentiation engine built on NumPy.

The package deliberately avoids a heavyweight deep-learning framework: the models
in :mod:`photocat.models` need only a small set of differentiable primitives
(linear layers, attention, layer normalisation, embeddings and graph message
passing).  Implementing them on a ~400 line autograd core keeps ``photocat``
installable with NumPy alone and makes every gradient auditable.

The engine uses reverse-mode differentiation over a dynamically built graph::

    x = Tensor([[1.0, 2.0]], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    x.grad  # [[2.0, 4.0]]

All computations are float64.  Parameters are plain :class:`Tensor` objects held
by the model; :class:`Adam` updates them in place.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Array = np.ndarray


def _unbroadcast(grad: Array, shape: tuple[int, ...]) -> Array:
    """Reduce ``grad`` back to ``shape`` after NumPy broadcasting."""
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A NumPy array participating in a differentiable computation graph."""

    __array_priority__ = 1000

    def __init__(
        self,
        data: Array | float | Sequence[float] | "Tensor",
        requires_grad: bool | None = None,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        if requires_grad is None:
            requires_grad = any(child.requires_grad for child in _children)
        self.requires_grad = bool(requires_grad)
        self.grad: Array | None = np.zeros_like(self.data) if self.requires_grad else None
        self._backward = lambda: None
        self._prev = tuple(child for child in _children if child.requires_grad)
        self._op = _op

    # ------------------------------------------------------------------ helpers
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    def __len__(self) -> int:
        return len(self.data)

    def detach(self) -> "Tensor":
        """Return a constant copy that is disconnected from the graph."""
        return Tensor(self.data.copy())

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad[...] = 0.0

    # ----------------------------------------------------------- arithmetic ops
    def __add__(self, other: "Tensor | float") -> "Tensor":
        other = as_tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="add")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        other = as_tensor(other)
        out = Tensor(self.data - other.data, _children=(self, other), _op="sub")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other: "Tensor | float") -> "Tensor":
        return as_tensor(other) - self

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        other = as_tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op="mul")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other: "Tensor | float") -> "Tensor":
        other = as_tensor(other)
        out = Tensor(self.data / other.data, _children=(self, other), _op="div")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                local = -out.grad * self.data / (other.data**2)
                other.grad += _unbroadcast(local, other.data.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other: "Tensor | float") -> "Tensor":
        return as_tensor(other) / self

    def __pow__(self, power: float) -> "Tensor":
        if isinstance(power, Tensor):
            raise TypeError("Tensor exponents are not supported")
        out = Tensor(self.data**power, _children=(self,), _op=f"pow{power}")

        def _backward() -> None:
            if self.requires_grad and power != 0:
                self.grad += out.grad * power * self.data ** (power - 1)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = as_tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op="matmul")

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += _unbroadcast(grad, self.data.shape)
            if other.requires_grad:
                grad = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += _unbroadcast(grad, other.data.shape)

        out._backward = _backward
        return out

    # -------------------------------------------------------------- reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _children=(self,), _op="sum")

        def _backward() -> None:
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                for ax in sorted(ax % self.ndim for ax in axes):
                    grad = np.expand_dims(grad, ax)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([self.data.shape[ax] for ax in axes]))
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), _children=(self,), _op="mean")

        def _backward() -> None:
            if not self.requires_grad:
                return
            grad = out.grad / count
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                for ax in sorted(ax % self.ndim for ax in axes):
                    grad = np.expand_dims(grad, ax)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def max(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        """Forward-only max, used for numerically stable softmax shifts."""
        return Tensor(self.data.max(axis=axis, keepdims=keepdims))

    # ---------------------------------------------------------------- reshaping
    def reshape(self, *shape: int | tuple[int, ...]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = tuple(int(s) for s in shape)  # type: ignore[arg-type]
        out = Tensor(self.data.reshape(new_shape), _children=(self,), _op="reshape")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        out = Tensor(self.data.transpose(axes), _children=(self,), _op="transpose")
        inverse = tuple(np.argsort(axes))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.transpose(inverse)

        out._backward = _backward
        return out

    def swapaxes(self, axis1: int, axis2: int) -> "Tensor":
        out = Tensor(self.data.swapaxes(axis1, axis2), _children=(self,), _op="swapaxes")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.swapaxes(axis1, axis2)

        out._backward = _backward
        return out

    # ------------------------------------------------------------- element-wise
    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), _children=(self,), _op="exp")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), _children=(self,), _op="log")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        return self**0.5

    def tanh(self) -> "Tensor":
        out = Tensor(np.tanh(self.data), _children=(self,), _op="tanh")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (1.0 - out.data**2)

        out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        out = Tensor(np.abs(self.data), _children=(self,), _op="abs")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * np.sign(self.data)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), _children=(self,), _op="relu")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0.0)

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """Tanh approximation of GELU (Hendrycks & Gimpel, 2016)."""
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * self * (1.0 + (c * (self + 0.044715 * self**3)).tanh())

    def sigmoid(self) -> "Tensor":
        # Stable logistic: 1 / (1 + exp(-x)).
        data = np.empty_like(self.data)
        positive = self.data >= 0
        data[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_x = np.exp(self.data[~positive])
        data[~positive] = exp_x / (1.0 + exp_x)
        out = Tensor(data, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * out.data * (1.0 - out.data)

        out._backward = _backward
        return out

    def masked_fill(self, mask: Array, value: float = -1e9) -> "Tensor":
        """Fill positions where ``mask`` is True with a constant (no gradient)."""
        mask = np.asarray(mask, dtype=bool)
        out = Tensor(np.where(mask, value, self.data), _children=(self,), _op="masked_fill")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += np.where(mask, 0.0, out.grad)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self - self.max(axis=axis, keepdims=True)
        exp = shifted.exp()
        return exp / exp.sum(axis=axis, keepdims=True)

    # ------------------------------------------------------------- graph seed
    def backward(self, grad: Array | None = None) -> None:
        """Backpropagate from this tensor (usually a scalar loss)."""
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor that does not require grad")
        order: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                visit(parent)
            order.append(node)

        visit(self)
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = np.asarray(grad, dtype=np.float64)
        for node in reversed(order):
            node._backward()


def as_tensor(value: Tensor | float | Array | Sequence[float]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(value)


def tensor(
    data: Array | float | Sequence[float], requires_grad: bool = False
) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def concat(tensors: Sequence[Tensor], axis: int = -1) -> Tensor:
    """Concatenate tensors along an existing axis."""
    if not tensors:
        raise ValueError("concat requires at least one tensor")
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, _children=tuple(tensors), _op="concat")

    def _backward() -> None:
        start = 0
        for item in tensors:
            if not item.requires_grad:
                start += item.data.shape[axis]
                continue
            slices = [slice(None)] * data.ndim
            slices[axis] = slice(start, start + item.data.shape[axis])
            item.grad += out.grad[tuple(slices)]
            start += item.data.shape[axis]

    out._backward = _backward
    return out


def embedding_lookup(weight: Tensor, indices: Array) -> Tensor:
    """Gather rows of ``weight``; the backward pass scatters gradients back."""
    idx = np.asarray(indices, dtype=np.int64)
    out = Tensor(weight.data[idx], _children=(weight,), _op="embedding")

    def _backward() -> None:
        if weight.requires_grad:
            np.add.at(weight.grad, idx, out.grad)

    out._backward = _backward
    return out


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalisation over the last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=-1, keepdims=True)
    return centered * (var + eps) ** -0.5 * gamma + beta


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Linear map ``x @ weight + bias`` with optional bias."""
    out = x @ weight
    if bias is not None:
        out = out + bias
    return out


def mse_loss(pred: Tensor, target: Array, weight: Array | None = None) -> Tensor:
    """(Optionally weighted) mean squared error."""
    diff = pred - target
    squared = diff * diff
    if weight is None:
        return squared.mean()
    return (squared * weight).sum() / weight.sum()


def bce_with_logits(logits: Tensor, target: Array, weight: Array | None = None) -> Tensor:
    """Numerically stable binary cross entropy from logits."""
    # max(x, 0) - x*y + log(1 + exp(-|x|))
    relu_part = logits.relu()
    linear_part = logits * target
    abs_logits = logits.abs()
    # log(1 + exp(-|x|)), evaluated without overflow.
    softplus = ((-abs_logits).exp() + 1.0).log()
    loss = relu_part - linear_part + softplus
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum()


class Adam:
    """AdamW-style optimiser (decoupled weight decay), in-place on ``Tensor``."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        unique: dict[int, Tensor] = {}
        for param in params:
            if param.requires_grad:
                unique[id(param)] = param
        self.params = list(unique.values())
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        self.t += 1
        beta1, beta2 = self.betas
        bias1 = 1.0 - beta1**self.t
        bias2 = 1.0 - beta2**self.t
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * param.data
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * (grad * grad)
            m_hat = self.m[i] / bias1
            v_hat = self.v[i] / bias2
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def clip_grad_norm(self, max_norm: float = 5.0) -> float:
        total = 0.0
        for param in self.params:
            if param.grad is not None:
                total += float(np.sum(param.grad * param.grad))
        norm = float(np.sqrt(total))
        if norm > max_norm and norm > 0:
            scale = max_norm / (norm + 1e-12)
            for param in self.params:
                if param.grad is not None:
                    param.grad *= scale
        return norm


def save_params(path: str, named_params: dict[str, Tensor]) -> None:
    np.savez_compressed(path, **{name: tensor_.data for name, tensor_ in named_params.items()})


def load_params(path: str, named_params: dict[str, Tensor]) -> None:
    with np.load(path) as archive:
        for name, tensor_ in named_params.items():
            if name not in archive:
                raise KeyError(f"parameter {name!r} missing from {path}")
            value = archive[name]
            if value.shape != tensor_.data.shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: checkpoint {value.shape} vs model {tensor_.data.shape}"
                )
            tensor_.data[...] = value
