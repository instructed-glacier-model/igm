from .line_search import LineSearch, ValueAndGradient
from .line_search_armijo import LineSearchArmijo
from .line_search_hager_zhang import LineSearchHagerZhang
from .line_search_wolfe import LineSearchWolfe

LineSearches = {
    "armijo": LineSearchArmijo,
    "hager-zhang": LineSearchHagerZhang,
    "wolfe": LineSearchWolfe,
}
