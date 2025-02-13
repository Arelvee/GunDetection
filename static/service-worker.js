self.addEventListener('push', function(event) {
    console.log('Push event received:', event);
    const payload = event.data ? event.data.text() : 'Default notification';

    const options = {
        body: payload,
        icon: 'static/notif.png', // Replace with the path to your notification icon
        vibrate: [200, 100, 200], // Vibration pattern (optional)
        // Adding sound notification
        sound: 'static/notif.mp3' // Replace with the path to your notification sound
    };

    event.waitUntil(
        self.registration.showNotification('New Notifications!', options)
    );
});
