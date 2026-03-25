const express = require('express');
const router = express.Router();
const { clerkWebhookHandler } = require('../controllers/webhookController');

router.post('/', clerkWebhookHandler);

module.exports = router;
