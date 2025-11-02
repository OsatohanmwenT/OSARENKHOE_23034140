"""
View database contents - Helper script
"""

from app import create_app
from app.database import EmotionRecord, User

app = create_app()

with app.app_context():
    print("\n" + "=" * 60)
    print("DATABASE CONTENTS")
    print("=" * 60)

    # Get all users
    users = User.query.all()
    print(f"\n📊 Total Users: {len(users)}")
    print("-" * 60)

    for user in users:
        print(f"\n👤 User ID: {user.id}")
        print(f"   Name: {user.name}")
        print(f"   Email: {user.email}")
        print(f"   Created: {user.created_at}")
        print(f"   Total Analyses: {len(user.emotions)}")

        # Show user's emotion records
        if user.emotions:
            print("\n   📝 Recent Analyses:")
            for record in user.emotions[:5]:  # Show last 5
                print(
                    f"      • {record.dominant_emotion} ({record.confidence:.1f}%) - {record.analyzed_at}"
                )

    # Get all emotion records
    records = EmotionRecord.query.all()
    print(f"\n\n📈 Total Emotion Records: {len(records)}")
    print("-" * 60)

    if records:
        print("\n🎭 Emotion Distribution:")
        emotions_count = {}
        for record in records:
            emotions_count[record.dominant_emotion] = (
                emotions_count.get(record.dominant_emotion, 0) + 1
            )

        for emotion, count in sorted(
            emotions_count.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / len(records)) * 100
            print(f"   {emotion}: {count} ({percentage:.1f}%)")

    print("\n" + "=" * 60 + "\n")
