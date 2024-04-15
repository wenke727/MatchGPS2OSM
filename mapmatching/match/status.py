from enum import IntEnum

class STATUS:
    FAILED       = -1 # 成功失败
    SUCCESS       = 0 # 成功匹配
    SAME_LINK     = 1 # 所有轨迹点位于同一条线上
    ONE_POINT     = 2 # 所有轨迹点位于同一个点上
    NO_CANDIDATES = 3 # 轨迹点无法映射到候选边上
    FAILED        = 4 # 匹配结果，prob低于阈值
    UNKNOWN       = 99

class CANDS_EDGE_TYPE:
    NORMAL = 0         # od 不一样
    ContinuousEdge = 1 # If points suggest the vehicle stays on the edge, od 位于同一条edge上，但起点相对终点位置偏前
    ReentryEdge  = 2 # the vehicle leaves and then re-enters the same edge
    ProximityEdge = 3 #