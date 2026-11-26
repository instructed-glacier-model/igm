from .line_search import LineSearch, ValueAndGradient
from .armijo import LineSearchArmijo
from .hager_zhang import LineSearchHagerZhang
from .wolfe import LineSearchWolfe

LineSearches = {
    "armijo": LineSearchArmijo,
    "hager-zhang": LineSearchHagerZhang,
    "wolfe": LineSearchWolfe,
}
