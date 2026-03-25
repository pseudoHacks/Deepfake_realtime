const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  clerkId: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  username: { type: String },
  photo: { type: String },
  firstName: { type: String },
  lastName: { type: String },
  role: { type: String, enum: ["admin", "member", "viewer"], default: "member" },
  availabilityStatus: { type: String, enum: ["active", "on_leave"], default: "active" }
}, { timestamps: true });

module.exports = mongoose.model('User', userSchema);
