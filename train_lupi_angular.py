import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import PKSutureDataset
from models import TeacherModel, StudentModel
from angular_labels import generate_angular_tension_labels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SECTORS = 12
EPOCHS = 10

def train():
    dataset = PKSutureDataset(r"C:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Sutures\DALK_SL_Pentacam")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    teacher = TeacherModel(NUM_SECTORS).to(DEVICE)
    student = StudentModel(NUM_SECTORS).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(teacher.parameters()) + list(student.parameters()),
        lr=1e-4
    )

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            slit = batch["slit"].to(DEVICE)
            axial = batch["axial"].to(DEVICE)

            # ---- Angular labels from axial map ----
            labels = []
            for i in range(axial.size(0)):
                axial_np = (axial[i].permute(1,2,0).cpu().numpy() * 255).astype("uint8")
                vec = generate_angular_tension_labels(axial_np, NUM_SECTORS)
                labels.append(vec)

            teacher_target = torch.tensor(labels, device=DEVICE)

            # ---- Teacher forward ----
            teacher_pred = teacher(slit, axial)

            # ---- Student forward ----
            student_pred = student(slit)

            # ---- Losses ----
            teacher_loss = F.mse_loss(teacher_pred, teacher_target)
            distill_loss = F.mse_loss(student_pred, teacher_pred.detach())

            loss = teacher_loss + distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

    torch.save(student.state_dict(), "student_angular_model.pth")
    print("✅ Student model saved")

if __name__ == "__main__":
    train()
