const { Webhook } = require('svix');
const User = require('../models/User');
const { clerkClient } = require('@clerk/clerk-sdk-node');

const clerkWebhookHandler = async (req, res) => {
  const WEBHOOK_SECRET = process.env.CLERK_WEBHOOK_SECRET;

  if (!WEBHOOK_SECRET) {
    console.error('Missing CLERK_WEBHOOK_SECRET');
    return res.status(500).json({ error: 'Missing Webhook Secret' });
  }

  // Get the headers and raw body
  const headers = req.headers;
  const payload = req.rawBody;

  // Get the Svix headers for verification
  const svix_id = headers['svix-id'];
  const svix_timestamp = headers['svix-timestamp'];
  const svix_signature = headers['svix-signature'];

  if (!svix_id || !svix_timestamp || !svix_signature) {
    return res.status(400).json({ error: 'Error occurred -- no svix headers' });
  }

  let evt;

  try {
    const wh = new Webhook(WEBHOOK_SECRET);
    evt = wh.verify(payload, {
      'svix-id': svix_id,
      'svix-timestamp': svix_timestamp,
      'svix-signature': svix_signature,
    });
  } catch (err) {
    console.error('Error verifying webhook:', err.message);
    return res.status(400).json({ error: 'Error verifying webhook' });
  }

  const { id } = evt.data;
  const eventType = evt.type;

  try {
    if (eventType === 'user.created') {
      const { email_addresses, username, first_name, last_name, image_url } = evt.data;
      const primaryEmail = email_addresses?.length > 0 ? email_addresses[0].email_address : '';

      try {
        await User.create({
          clerkId: id,
          email: primaryEmail,
          username: username || '',
          firstName: first_name || '',
          lastName: last_name || '',
          photo: image_url || '',
        });
      } catch (error) {
        if (error.code === 11000) {
          console.warn(`User with clerkId ${id} or email ${primaryEmail} already exists.`);
        } else {
          throw error;
        }
      }
    }

    if (eventType === 'user.updated') {
      const { email_addresses, username, first_name, last_name, image_url } = evt.data;
      const primaryEmail = email_addresses?.length > 0 ? email_addresses[0].email_address : '';

      await User.findOneAndUpdate(
        { clerkId: id },
        {
          email: primaryEmail,
          username: username || '',
          firstName: first_name || '',
          lastName: last_name || '',
          photo: image_url || '',
        },
        { new: true }
      );
    }

    if (eventType === 'user.deleted') {
      await User.findOneAndDelete({ clerkId: id });
    }

    if (eventType === 'organizationMembership.deleted') {
      const { public_user_data } = evt.data;
      const userId = public_user_data?.user_id;

      if (userId) {
        // Downgrade role in DB
        await User.findOneAndUpdate(
          { clerkId: userId },
          { role: 'member' }
        );

        // Optionally update user metadata in Clerk
        await clerkClient.users.updateUserMetadata(userId, {
          publicMetadata: {
            role: 'member'
          }
        });
      }
    }

    res.status(200).json({ success: true, message: 'Webhook received and processed' });
  } catch (error) {
    console.error(`Error processing webhook event ${eventType}:`, error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
};

module.exports = {
  clerkWebhookHandler
};
