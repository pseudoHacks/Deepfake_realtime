const { ClerkExpressRequireAuth } = require('@clerk/clerk-sdk-node');

const requireAuth = ClerkExpressRequireAuth({});

const requireRole = (allowedRoles) => {
  return (req, res, next) => {
    try {
      const userRole = req.auth?.sessionClaims?.o?.rol;

      if (!userRole || !allowedRoles.includes(userRole)) {
        return res.status(403).json({ error: 'Access denied: insufficient permissions' });
      }

      next();
    } catch (error) {
      console.error("Auth Middleware Error:", error);
      res.status(403).json({ error: 'Access denied' });
    }
  };
};

module.exports = {
  requireAuth,
  requireRole
};
