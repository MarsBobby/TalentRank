#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cache.py — Redis 缓存层
────────────────────────
职责分工：
  1. 热点评分缓存   — analyze 结果缓存 5 分钟（TTL=300s）
  2. Session 向量   — 替代内存 _session_vecs，TTL=1800s（30分钟）
  3. Bandit 权重    — 跨进程共享最新权重快照
  4. 限流计数器     — 防止短时间内重复提交大量文件
"""

import os
import json
import pickle
import logging
from typing import Any, Optional, Dict, List
from datetime import timedelta

logger = logging.getLogger(__name__)

# ── 连接配置 ──────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# TTL 常量（秒）
TTL_ANALYZE_RESULT = 300       # 分析结果缓存 5 分钟
TTL_SESSION_VEC    = 1800      # 候选人向量 30 分钟（供 Bandit 反馈）
TTL_BANDIT_WEIGHT  = 86400     # Bandit 权重 24 小时
TTL_RATE_LIMIT     = 60        # 限流窗口 60 秒

# Key 前缀
PFX_ANALYZE   = "tr:analyze:"   # tr:analyze:{function}:{files_hash}
PFX_SESSION   = "tr:session:"   # tr:session:{function}:{candidate_id}
PFX_BANDIT    = "tr:bandit:"    # tr:bandit:{function}
PFX_RATE      = "tr:rate:"      # tr:rate:{client_ip}


class CacheClient:
    """Redis 缓存客户端（懒加载，连接失败时降级为 no-op）"""

    def __init__(self):
        self._client = None
        self._available = False
        self._connect()

    def _connect(self):
        try:
            import redis
            self._client = redis.from_url(
                REDIS_URL,
                decode_responses=False,   # 允许存储二进制（pickle）
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True,
            )
            self._client.ping()
            self._available = True
            logger.info(f"Redis 连接成功: {REDIS_URL}")
        except Exception as e:
            self._available = False
            logger.warning(f"Redis 连接失败，缓存层降级为 no-op: {e}")

    @property
    def available(self) -> bool:
        return self._available

    # ─── 通用 Get/Set ────────────────────────────────────────

    def get_json(self, key: str) -> Optional[Any]:
        """读取 JSON 数据"""
        if not self._available:
            return None
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.debug(f"Cache GET失败 [{key}]: {e}")
            return None

    def set_json(self, key: str, value: Any, ttl: int = 300) -> bool:
        """写入 JSON 数据"""
        if not self._available:
            return False
        try:
            self._client.setex(key, ttl, json.dumps(value, ensure_ascii=False, default=str))
            return True
        except Exception as e:
            logger.debug(f"Cache SET失败 [{key}]: {e}")
            return False

    def get_pickle(self, key: str) -> Optional[Any]:
        """读取 Pickle 对象（用于 Python 对象如特征向量）"""
        if not self._available:
            return None
        try:
            raw = self._client.get(key)
            return pickle.loads(raw) if raw else None
        except Exception as e:
            logger.debug(f"Cache GET(pickle)失败 [{key}]: {e}")
            return None

    def set_pickle(self, key: str, value: Any, ttl: int = 1800) -> bool:
        """写入 Pickle 对象"""
        if not self._available:
            return False
        try:
            self._client.setex(key, ttl, pickle.dumps(value))
            return True
        except Exception as e:
            logger.debug(f"Cache SET(pickle)失败 [{key}]: {e}")
            return False

    def delete(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception:
            return False

    def delete_pattern(self, pattern: str) -> int:
        """删除匹配 pattern 的所有 Key（SCAN 实现，生产安全）"""
        if not self._available:
            return 0
        count = 0
        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    self._client.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.debug(f"Cache DELETE PATTERN失败 [{pattern}]: {e}")
        return count

    # ─── 业务级封装 ──────────────────────────────────────────

    def get_analyze_result(self, function: str, files_hash: str) -> Optional[Dict]:
        """读取分析结果缓存"""
        key = f"{PFX_ANALYZE}{function}:{files_hash}"
        result = self.get_json(key)
        if result:
            logger.info(f"Cache HIT: analyze [{function}]")
        return result

    def set_analyze_result(self, function: str, files_hash: str, result: Dict) -> bool:
        """写入分析结果缓存（TTL=5分钟）"""
        key = f"{PFX_ANALYZE}{function}:{files_hash}"
        ok = self.set_json(key, result, TTL_ANALYZE_RESULT)
        if ok:
            logger.info(f"Cache SET: analyze [{function}] TTL={TTL_ANALYZE_RESULT}s")
        return ok

    def get_session_vec(self, function: str, candidate_id: str) -> Optional[Any]:
        """读取候选人向量（Bandit 反馈用）"""
        key = f"{PFX_SESSION}{function}:{candidate_id}"
        return self.get_pickle(key)

    def set_session_vecs(self, function: str, vecs: list) -> None:
        """批量写入候选人向量"""
        for vec in vecs:
            key = f"{PFX_SESSION}{function}:{vec.candidate_id}"
            self.set_pickle(key, vec, TTL_SESSION_VEC)
        logger.info(f"Cache SET: {len(vecs)} session vecs [{function}]")

    def get_bandit_weights(self, function: str) -> Optional[Dict]:
        """读取 Bandit 最新权重"""
        key = f"{PFX_BANDIT}{function}"
        return self.get_json(key)

    def set_bandit_weights(self, function: str, weights: Dict,
                            update_count: int = 0) -> bool:
        """写入 Bandit 权重（24小时）"""
        key = f"{PFX_BANDIT}{function}"
        payload = {"weights": weights, "update_count": update_count,
                   "updated_at": __import__("datetime").datetime.utcnow().isoformat()}
        return self.set_json(key, payload, TTL_BANDIT_WEIGHT)

    def check_rate_limit(self, client_key: str, max_requests: int = 10) -> bool:
        """
        限流检查（滑动窗口）
        返回 True = 允许通过, False = 超过限制
        """
        if not self._available:
            return True
        key = f"{PFX_RATE}{client_key}"
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, TTL_RATE_LIMIT)
            results = pipe.execute()
            current = results[0]
            if current > max_requests:
                logger.warning(f"Rate limit exceeded: {client_key} ({current}/{max_requests})")
                return False
            return True
        except Exception:
            return True  # Redis 异常时放行

    def flush_function_cache(self, function: str) -> None:
        """清空某职能的所有分析缓存（重新分析时调用）"""
        self.delete_pattern(f"{PFX_ANALYZE}{function}:*")
        self.delete_pattern(f"{PFX_SESSION}{function}:*")
        logger.info(f"Cache FLUSH: [{function}]")

    def health_check(self) -> Dict:
        """Redis 健康检查"""
        if not self._available:
            return {"status": "unavailable", "mode": "no-op fallback"}
        try:
            info = self._client.info("server")
            return {
                "status": "ok",
                "version": info.get("redis_version", "?"),
                "used_memory_human": info.get("used_memory_human", "?"),
                "connected_clients": info.get("connected_clients", "?"),
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}


# ── 计算文件列表的哈希（作为缓存 Key 的一部分） ───────────────

def compute_files_hash(filenames: List[str], file_sizes: List[int]) -> str:
    """根据文件名+大小生成轻量哈希（不读文件内容，O(1)）"""
    import hashlib
    combined = "|".join(
        f"{n}:{s}" for n, s in sorted(zip(filenames, file_sizes))
    )
    return hashlib.md5(combined.encode()).hexdigest()[:12]


# ── 全局单例 ──────────────────────────────────────────────────
cache = CacheClient()
